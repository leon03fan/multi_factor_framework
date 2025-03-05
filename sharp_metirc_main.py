import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from gplearn.genetic import SymbolicTransformer # 符号变换
from gplearn.fitness import make_fitness
from gplearn.functions import _function_map

from sharpe_metric import  sharpe_metric
import comm as comm


if __name__ == '__main__':
	start_time = time.time()

	g_path_in_name = "./in"
	g_path_out_name = "./out"
	g_file_name = "大商所-豆一-a00.DF"
	g_str_freq =  "15m"
	g_full_file_name = f"{g_path_in_name}/{g_file_name}-none-{g_str_freq}.pkl"

	date_begin = datetime(2020, 1, 1)
	my_featurs = ['open', 'high', 'low', 'close', 'volume']
	# 后续要添加因子生成的features

	nd_train_X, nd_train_y, _, _, date_split= comm.split_X_y(p_file_path= g_full_file_name, p_begin_date=date_begin, p_lst_feature=my_featurs)
	# 后续要添加因子生成的features

	my_func = list(_function_map.keys()) # [:10]
	my_metric = make_fitness(function=sharpe_metric, greater_is_better=True, wrap=False)

	ST_gplearn = SymbolicTransformer(
									population_size=100000,     # 一次生成因子的数量，
	                                 hall_of_fame=800,          #
	                                 n_components=800,          # 最终输出多少个因子
	                                 generations=3,             # 非常非常非常重要！！！--进化多少轮次？3也就顶天了
	                                 tournament_size=800,       # 锦标赛入选的因子数量
	                                 const_range=None,          # 常数取值范围 (-1, 1),  # critical
	                                 init_depth=(2, 3),         # 第二重要的一个部位，控制我们公式的一个深度
	                                 #  function_set=user_func, # 输入的算子群
	                                 function_set=my_func,      # 输入的算子群
	                                 metric=my_metric,          # 提升的点
	                                 # metric='pearson',        # pearson相关系数
	                                 parsimony_coefficient=0.001,
	                                 p_crossover=0.4,
	                                 p_subtree_mutation=0.02,
	                                 p_hoist_mutation=0.02,
	                                 p_point_mutation=0.02,
	                                 p_point_replace=0.45,
	                                 feature_names=my_featurs,	# 注意这里必须有feature_names 把重要算子“聚合化”，
	                                 n_jobs=1,
	                                 random_state=1)
	ST_gplearn.fit(nd_train_X, nd_train_y)

	best_programs = ST_gplearn._best_programs
	best_programs_dict = {}

	for bp in best_programs:
		factor_name = 'alpha_' + str(best_programs.index(bp) + 1)
		best_programs_dict[factor_name] = {'fitness': bp.fitness_, 'expression': str(bp), 'depth': bp.depth_, 'length': bp.length_}

	best_programs_frame = pd.DataFrame(best_programs_dict).T
	best_programs_frame = best_programs_frame.sort_values(by='fitness', axis=0, ascending=False)
	best_programs_frame = best_programs_frame.drop_duplicates(subset=['expression'], keep='first')

	print(best_programs_frame)
	best_programs_frame.to_csv(f'{g_path_out_name}/score-{g_file_name}-{g_str_freq}.csv')



	end_time = time.time()
	print('Time Cost:-----    ', end_time - start_time, 'S    --------------')