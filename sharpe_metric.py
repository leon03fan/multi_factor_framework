from numpy.ma.core import cumprod
from sklearn.linear_model import LinearRegression

import numpy as np
import  pandas as pd
from comm import comm


def fit_metric(y, y_pred, w):
	"""
	夏普比率
	Args:
		y: 实际收益率
		y_pred: 预测收益率
		w: 权重
	Returns:
		float: 夏普比率
	"""
	# 如果预测值太少 返回0
	if len(y_pred) < 100:
		return 0
	# 读取数据集 并初始化
	sm 						= SharpeMetric()
	sm.prepare_data()
	# 拟合
	ndarray_x_train 		= y_pred.reshape(-1, 1)
	ndarray_y_train 		= sm.ndarray_price_change
	linear_model 			= LinearRegression(fit_intercept=True) # 这里应该考虑是否需要fit_intercept
	linear_model.fit(ndarray_x_train, ndarray_y_train)
	# 计算y_hat
	ndarray_gp_y_train_hat 	= linear_model.predict(ndarray_x_train)
	ndarray_gp_y_train_hat	= ndarray_gp_y_train_hat.reshape(-1)
	# 这里不再对y_hat进行滚动标准化 如果进行滚动标准化 会改变预测值，如本来是负值变为正值
	# 同时，在functions.py中，已经对y_hat进行了滚动标准化
	# ndarray_y_hat 			= sm.rolling_norm(ndarray_y_hat) 
	# 计算基础信息：净值，收益率，其中考虑到了手续费
	sm.calc_position_and_nav(ndarray_gp_y_train_hat)
	# 计算夏普比率
	sharpe_ratio 			= sm.calc_sharpe_ratio()
	return sharpe_ratio


class SharpeMetric:
	def __init__(self):
		"""
		初始化 SharpeMetric
		"""
		obj_comm = comm()
		self.split_train_file 		= f"{obj_comm.path_split_date}/{obj_comm.file_name}-none-{obj_comm.str_freq}-train.pkl"		# bar数据文件路径

		self.n_of_year 				= obj_comm.n_of_year	# 年化周期数
		self.rf 					= obj_comm.rf			# 无风险利率
		self.predict_n 				= obj_comm.predict_n	# 预测多少根Bar
		self.fee_rate 				= obj_comm.fee_rate		# 交易费率
		self.pos_coef 				= obj_comm.pos_coef		# 仓位系数
		self.pos_thd 				= obj_comm.pos_thd		# 仓位阈值

		self.ndarray_close 			= np.array([])			# close
		self.ndarray_tdate 			= np.array([])			# tdate
		self.ndarray_price_change 	= np.array([])			# 价格变化

		self.ndarray_position 		= np.array([])			# position
		self.ndarray_ret 			= np.array([])			# ret
		self.ndarray_nav 			= np.array([])			# nav

	def prepare_data(self):
		"""读取前面处理过的 保存在split_data目录下的Train数据集
		Returns:
			_type_: DataFrame
		"""
		tmp_df_train				= pd.read_pickle(self.split_train_file)
		self.ndarray_close 			= tmp_df_train["close"].values
		self.ndarray_tdate 			= tmp_df_train["tdate"].values
		self.ndarray_price_change 	= tmp_df_train["price_change"].values

	def calc_position_and_nav(self, p_ndarray_gp_y_hat: np.ndarray):
		"""
		计算仓位、手续费、净值：
		净值(长度比list_close少一位)
		Args:
			p_df_train: 数据集,包含index,date,close,ret,y_hat_norm
		"""
		# 计算每根ba预测仓位
		# 后续可以用sigmoid、tanh、relu，zscore等转换
		self.ndarray_position 		= p_ndarray_gp_y_hat / 0.0005 * self.pos_coef
		# 计算策略收益率
		self.ndarray_ret 			= self.ndarray_position[:-1] * self.ndarray_price_change[1:]
		# 计算手续费	
		temp_ndarray_strategy_fee	= np.abs(self.ndarray_position[1:] - self.ndarray_position[:-1]) * self.fee_rate
		# temp_ndarray_strategy_fee	= 0
		self.ndarray_nav 			= 1 + (self.ndarray_ret - temp_ndarray_strategy_fee).cumsum()

	def calc_sharpe_ratio(self):
		"""
		计算夏普比率
		Returns:
			float: 夏普比率
		"""
		# 年化收益率
		int_days 					= len(np.unique(self.ndarray_tdate))
		annual_return 				= self.ndarray_nav[-1] ** (252/int_days) - 1
		# 获得日最后的K线累计净值
		temp_unique_dates 			= np.unique(self.ndarray_tdate)
		temp_last_indices 			= np.searchsorted(self.ndarray_tdate, temp_unique_dates, side='right') - 1
		temp_last_indices[-1] 		= temp_last_indices[-1] - 1  # 最后一个bar的没有净值 所以减一
		temp_ndarray_day_last_nav 	= self.ndarray_nav[temp_last_indices]
		# 计算日收益率
		temp_nd_daily_returns 		= np.diff(temp_ndarray_day_last_nav) / temp_ndarray_day_last_nav[:-1]
		# 计算年化收益波动率
		annual_date_std 			= temp_nd_daily_returns.std() * np.sqrt(self.n_of_year)

		# 夏普比率
		if self.ndarray_nav[-1] < 0:
			print(f"全部赔完了，夏普比率为0")
		elif annual_date_std > 0 and not np.isnan(annual_return):
			sharpe = (annual_return - self.rf) / annual_date_std
		else:
			sharpe = 0
		
		return sharpe

	def acum_norm(self, p_nd):
		"""
		累计归一化
		Args:
			p_nd: 一列ndarray，用于后续做归一化操作的数据
		Returns:
			ndarray: 归一化后的ndarray
		"""
		# 合并数据集
		factors_data 	= pd.DataFrame(p_nd, columns=["factor"])
		factors_data 	= factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
		# 计算中间变量
		factors_mean 	= factors_data.cumsum() / np.arange(1, len(factors_data) + 1)[:,np.newaxis]
		factors_std 	= factors_data.expanding().std()
		# 计算最终结果
		factor_value 	= (factors_data - factors_mean) / factors_std
		factor_value 	= factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
		factor_value 	= factor_value.clip(-self.pos_thd, self.pos_thd)

		x = np.nan_to_num(factor_value['factor'].values)
		return x

	def rolling_norm(self, x, window=2000):
		"""
		滚动归一化
		Args:
			x: 一列ndarray，用于后续做归一化操作的数据
			window: 滚动窗口大小
		Returns:
			ndarray: 归一化后的ndarray
		"""
		# 合并数据集
		x 				= x.replace([np.inf, -np.inf, np.nan], 0.0)
		# 计算中间变量
		factors_mean 	= x.rolling(window=window, min_periods=1).mean()
		factors_std 	= x.rolling(window=window, min_periods=1).std()
		# 计算最终结果
		factor_value 	= (x - factors_mean) / factors_std
		factor_value 	= factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
		factor_value 	= factor_value.clip(-self.pos_thd, self.pos_thd)
		return np.nan_to_num(factor_value.values)


