from numpy.ma.core import cumprod
from sklearn.linear_model import LinearRegression

import numpy as np
import  pandas as pd
import comm as comm

class SharpeMetric:
	def __init__(self, p_n_of_year:int=252, p_rf:float=0.03, p_predict_n:int=1, p_fee_rate:float=0.0002, p_pos_coef:float=1, p_pos_thd:float=1):
		"""
		初始化 SharpeMetric
		Args:
			p_file_path (str): bar数据文件路径
			p_n_of_year (int, optional): 年交易天数 Defaults to 252.
			p_rf (float, optional): 无风险利率 Defaults to 0.03.
			p_predict_n (int, optional): 预测多少根Bar Defaults to 1.
			p_fee_rate (float, optional): 交易费率 Defaults to 0.0002.
			p_pos_coef (float, optional): 仓位系数 Defaults to 1.
			p_pos_thd (float, optional): 仓位阈值 Defaults to 1.
		"""
		obj_comm = comm()
		self.p_file_path = f"{obj_comm.path_split_date}/{obj_comm.file_name}-none-{obj_comm.str_freq}-train.pkl"		# bar数据文件路径
		self.p_n_of_year = p_n_of_year		# 年化周期数
		self.p_rf = p_rf					# 无风险利率
		self.p_predict_n = p_predict_n		# 预测多少根Bar
		self.p_fee_rate = p_fee_rate		# 交易费率
		self.p_pos_coef = p_pos_coef		# 仓位系数
		self.p_pos_thd = p_pos_thd			# 仓位阈值

	def pre_data(self):
		"""读取前面处理过的 保存在split_data目录下的Train数据集
		Returns:
			_type_: DataFrame
		"""
		df_train= pd.read_pickle(self.p_file_path)
		df_train = df_train[["tdate","close","ret"]]    # 只取收盘价、收益率、时间戳、日期
		return df_train

	def  calc_base_param(self, p_df_train):
		"""
		回测数据集处理
		计算相关指标：
		Args:
			p_df_train: 数据集,包含index,date,close,ret,y_hat_norm
		Returns:
			ndarray: 净值
		"""
		df_temp = p_df_train.copy()
		# 计算价格每根bar的每日涨跌幅
		# df_temp["price_change_pct"] = df_temp["close"].pct_change(1).fillna(0)

		# 计算每根ba预测仓位
		df_temp["position"] = df_temp["y_hat"] / 0.0005

		# 计算净值
		arr_predict_postion = df_temp["position"].values
		arr_price_change = df_temp["ret"].values
		arr_strategy_returns = arr_predict_postion[:-1] * arr_price_change[1:]
		arr_strategy_fee = np.abs(arr_predict_postion[1:] - arr_predict_postion[:-1]) * self.p_fee_rate

		arr_strategy_nav = 1 + (arr_strategy_returns - arr_strategy_fee).cumsum()
		df_temp = df_temp.iloc[:-1]
		df_temp["nav"] = arr_strategy_nav
		return df_temp

	def calc_sharpe_ratio(self, p_df_data:pd.DataFrame):
		"""
		计算夏普比率
		Args:
			df_result: 数据集
		Returns:
			float: 夏普比率
		"""
        # 年化净值收益率
		int_days = len(list(set(p_df_data["tdate"])))
		arr_nav = p_df_data["nav"].values
		annual_return = (arr_nav[-1] ** (252/int_days)) - 1 
		# 将15分钟K线ret 转化为日ret 然后计算年化收益波动率
		arr_daily_returns = p_df_data.groupby("tdate").last()["nav"].pct_change().replace([np.inf, -np.inf, np.nan], 0.0).values
		annual_date_std = arr_daily_returns.std() * np.sqrt(self.p_n_of_year)
		# 夏普比率
		sharpe = (annual_return - self.p_rf) / annual_date_std if (annual_date_std != 0 and np.isnan(annual_return)) else 0 # 夏普比率
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
		factors_data = pd.DataFrame(p_nd, columns=["factor"])
		factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
		# 计算中间变量
		factors_mean = factors_data.cumsum() / np.arange(1, len(factors_data) + 1)[:,np.newaxis]
		factors_std = factors_data.expanding().std()
		# 计算最终结果
		factor_value = factors_data - factors_mean / factors_std
		factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
		factor_value = factor_value.clip(-self.p_pos_thd, self.p_pos_thd)

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
		# 最小要2个月的数据 后面加上判断
		# 合并数据集
		factors_data = pd.DataFrame(x, columns=['factor'])
		factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
		# 计算中间变量
		factors_mean = factors_data['factor'].rolling(window=window, min_periods=1).mean()
		factors_std = factors_data['factor'].rolling(window=window, min_periods=1).std()
		# 计算最终结果
		factor_value = (factors_data['factor'] - factors_mean) / factors_std
		factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
		factor_value = factor_value.clip(-self.p_pos_thd, self.p_pos_thd)
		x = np.nan_to_num(factor_value.values)
		return x

def sharpe_metric(y, y_pred, w):
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
	if len(y_pred) < 10:
		return 0
	# 读取数据集 并初始化
	sm = SharpeMetric()
	df_train = sm.pre_data().copy()
	# 将预测值拼接到数据集中
	df_factor =pd.DataFrame(y_pred, columns=["factor"])
	df_train["factor"] = y_pred
	# df_train = pd.concat([df_train, df_factor], ignore_index=False, axis=1)
	df_train = df_train.replace([np.inf, -np.inf, np.nan], 0.0)
	# 拟合
	nd_x_train = df_factor['factor'].values.reshape(-1,1)
	nd_y_train = df_train['ret'].values.reshape(-1,1)
	linear_model = LinearRegression(fit_intercept=True)
	linear_model.fit(nd_x_train, nd_y_train)
	# 计算y_hat
	y_train_hat = linear_model.predict(nd_x_train)
	df_train["y_hat"] = [i[0] for i in y_train_hat]
	# df_train["position"] =sm.rolling_norm(df_train['y_hat'].values) # 这里不应该再对y_hat进行滚动标准化 如果进行滚动标准化 会改变预测值，如本来是负值变为正值
	# 计算基础信息：净值，收益率，其中考虑到了手续费
	df_train = sm.calc_base_param(df_train)
	# 计算夏普比率
	sharpe_ratio = sm.calc_sharpe_ratio(df_train)
	return sharpe_ratio

