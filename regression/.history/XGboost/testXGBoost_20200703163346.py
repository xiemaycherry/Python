import xgboost as xgb
from xgboost import XGBRegressor

import pandas as pd 
import numpy as np 

from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn.datasets import load_boston

import warnings
warnings.filterwarnings('ingore')

# download data set
X, y = load_boston(return_X_y = True)