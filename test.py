import numpy as np
import pandas as pd
import pickle

# df = pd.read_pickle("./split_data/大商所-豆一-a00.DF-none-5m-train.pkl")
# print(df.head())

# 请帮我写一个例子，展示df_train['close'].pct_change(periods=p_n).shift(-p_n)的效果

df = pd.read_pickle("./split_data/大商所-豆一-a00.DF-none-5m-train.pkl")

df['ret'] = df['close'].pct_change(periods=5)
df['ret_shift'] = df['ret'].shift(-5)

print(df[['close', 'ret', 'ret_shift']].head(20))

