import numpy as np
import pandas as pd
import os
from datetime import datetime
import configparser


class comm:
	def __init__(self):
		"""设置公用参数 后续可以修改为从数据库读取 
		"""
		config_file = 'config.ini'
		config = configparser.ConfigParser(inline_comment_prefixes=(';',))
		config.read(config_file, encoding='utf-8')
		
		# 读取配置
		self.path_in_name 		= config.get('Paths', 'input_path')
		self.path_out_name 		= config.get('Paths', 'output_path')
		self.path_split_date 	= config.get('Paths', 'split_data_path')

		self.file_name 			= config.get('File', 'file_name')
		self.str_freq 			= config.get('File', 'frequency')

		self.n_of_year			= int(config.get('Data', 'n_of_year'))
		self.rf					= float(config.get('Data', 'rf'))
		self.predict_n			= int(config.get('Data', 'predict_n'))
		self.fee_rate			= float(config.get('Data', 'fee_rate'))
		self.pos_coef			= float(config.get('Data', 'pos_coef'))
		self.pos_thd			= float(config.get('Data', 'pos_thd'))

	def split_X_y(self, p_file_path, p_begin_date:str = None, p_end_date:str = None, 
			   p_lst_feature:str = ["open", "high", "low", "close", "volume"]):
		"""
		将数据集切分为X和y
		Args:
			p_file_path:数据集全路径文件名
			p_begin_date:训练集开始日期
			p_end_date:训练集结束日期
			p_lst_feature:特征列表
			p_n:预测多少根Bar
		Returns:
			ndarray: X 训练数据集
			ndarray: y 训练数据集
			pd.DataFrame: df_train 训练数据集
			pd.DataFrame: df_test 测试数据集
			datetime: date_split 切分点日期
		"""
		p_n				= self.predict_n
		# 读取数据集
		df_all 			= pd.read_pickle(p_file_path)
		df_all.index 	= pd.to_datetime(df_all.index)

		if p_begin_date is not None:
			date_begin 	= pd.to_datetime(p_begin_date)
			df_all 		= df_all[df_all.index >= date_begin]
		df_all = df_all.sort_index()

		if p_end_date is not None:
			date_split 	= pd.to_datetime(p_end_date)
		else:
			# 按照df_all的index的顺序，根据记录条数，按照8:2的比例进行切分，找到切分点的那条记录的index
			int_total_records = len(df_all)
			int_split_point = int(int_total_records * 0.8)
			date_split = pd.to_datetime(df_all.index[int_split_point].date())  # 直接取index值

		# 根据idx_split，将df_all切分为df_train和df_oot
		df_train 	= df_all[df_all.index <= date_split]
		df_test 	= df_all[df_all.index > date_split]

		# 这里后续要添加产品信息和切割点到数据库，以便后续再次读取操作

		# # 计算收益率
		# df_train['price_change'] = df_train['close'].pct_change(periods=p_n).shift(-p_n) # 计算收益率 用于后续拟合Y
		df_train["price_change"] = (df_train["close"].shift(-p_n) - df_train["close"])/ df_train["close"]# 计算收益率 用于后续拟合Y
		df_train.dropna(axis=0, how='any', inplace=True)
		# 计算日期
		df_train["tdate"] = pd.to_datetime(df_train.index).date
		# 获取建模用数据
		np_train_x = df_train[p_lst_feature].to_numpy()
		np_train_y = df_train['price_change'].values.reshape(-1,1)
		# 保存df_train和df_test
		temp_file_name = p_file_path.split("/")[-1][:-4]
		df_train.to_pickle(f"./split_data/{temp_file_name}-train.pkl")
		df_test.to_pickle(f"./split_data/{temp_file_name}-test.pkl")

		return np_train_x, np_train_y, df_train, df_test, date_split


