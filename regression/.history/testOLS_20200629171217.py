print(__doc__)

import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score 

# Load the diabetes dataset
diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y= True)

# select one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data set into training/testing sets
X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
