import xgboost as xgb
from xgboost import XGBRegressor

import pandas as pd 
import numpy as np 

from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn.datasets import load_boston

import warnings
warnings.filterwarnings('ignore')

# download data set
X, y = load_boston(return_X_y = True)
print(X.shape)
# split train-test
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 2)
# Model : xgboost 自定义了有个数据矩阵类DMatrix, 在训练的开始时进行一篇预处理，
