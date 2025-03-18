import numpy as np
import pandas as pd
import pickle

import pandas as pd
import numpy as np

import numpy as np

# 假设 temp_ndarray_day_last_nav 是一个 numpy 数组
temp_ndarray_day_last_nav = np.array([100, 101, 102, 103])
print(temp_ndarray_day_last_nav)

# 计算百分比变化
pct_change = np.diff(temp_ndarray_day_last_nav) / temp_ndarray_day_last_nav[:-1]
print(np.diff(temp_ndarray_day_last_nav))
print(pct_change)

# 在开头添加 NaN，以匹配 pct_change() 的结果
pct_change = np.insert(pct_change, 0, np.nan)
print(pct_change)