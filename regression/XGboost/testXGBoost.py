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
xgb_train = xgb.DMatrix(x_train, y_train)
xgb_test = xgb.DMatrix(x_test, y_test)

# parameters
params = {
    'booster': 'gbtree',
    'objective':  # 多分类问题
    'num_class': # 类别数目， multisoftmax
    'gamma': # 损失下降多少才进行分裂
    'silent': '0', # control output information
    'nthread': 7, # cpu 默认最大
    'eta': 0.7, # learning rate
    'min_child_weight': 3, # 参数默认是1， 
    'seed': 
    'colsampe_bytree': # 生成树进行的列采样
}