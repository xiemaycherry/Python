---
title: scikit-learn
date: 2019-03-23 18:39:47
tags: Python
categories: [Python]
---

<!-- more -->

# Day Ox00 

## python 组织

## 包
包是一个有层次的文件目录结构，它定义了由n个模块或n个子包组成的python应用程序执行环境。通俗一点：包是一个包含__init__.py 文件的目录，该目录下一定得有这个__init__.py文件和其它模块或子包。

__init__.py可以是空文件，在此处reshape包下的这个文件就是空的。当然，也可以有Python代码，因为__init__.py本身就是一个模块。

### Python 模块

Python 模块(Module)，是一个 Python 文件，以 .py 结尾，包含了 Python 对象定义和Python语句。

**注意！** 导入一个模块，使用import！

```python
import A # A模块名
from A import B # A为模块名，B为模块A中的某个类、方法或者变量等
from A imprt * # 导入模块A的所以内容
```



## 库
库是指具有相关功能模块的集合。这也是Python的一大特色之一，即具有强大的标准库、第三方库以及自定义模块。

标准库：python里那些自带的模块

第三方库：就是由其他的第三方机构，发布的具有特定功能的模块。比如2020年十大最受欢迎库：TensorFlow、Scikit-Learn、Numpy、Keras、PyTorch、LightGBM、Eli5、SciPy、Theano、Pandas

python标准库和第三方库的区别

python的标准库是随着pyhon安装的时候默认自带的库。

python的第三方库，需要下载后安装到python的安装目录下，不同的第三方库安装及使用方法不同。

它们调用方式是一样的，都需要用import语句调用。

## sklearn的组织

sklearn是基于numpy和scipy的一个机器学习算法库

# sklearn的通用模式

数据加载；划分数据集；建立模型；训练模型；模型评估

```python
#导入模块
from sklearn.model_selection import train_test_split
from sklearn import datasets
#k近邻函数
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
#导入数据和标签
iris_X = iris.data
iris_y = iris.target
#划分为训练集和测试集数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
#print(y_train)
#设置knn分类器
knn = KNeighborsClassifier()
#进行训练
knn.fit(X_train,y_train)
#使用训练好的knn进行数据预测
print(knn.predict(X_test))
print(y_test)
```

## 常用的自带数据库

```python
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data) # feature
digits.target
X, y = datasets.load_iris(return_X_y=True)
```



# Cross-validation: evaluating estimator performance[¶](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance)

![Grid Search Workflow](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

```
import numpy as np
from sklearn.model_selection import train_test_split

# 调用train_test_split函数 自动划分数据集 40%for testing
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target, test_size=0.4, random_state=0)
```



## corss validation

![../_images/grid_search_cross_validation.png](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

```
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                        cv=5, return_train_score=False)
sorted(scores.keys())
```

## cross-validation metrics

```
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel = 'linear', C = 1)
```



## Cross validation of time series data

![../_images/sphx_glr_plot_cv_indices_0101.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_0101.png)

# Tuning the hyper-parameters of an estimator

A search consists of:

- an estimator (regressor or classifier such as `sklearn.svm.SVC()`);

- a parameter space;

- a method for searching or sampling candidates;

- a cross-validation scheme; and

- a [score function](https://scikit-learn.org/stable/modules/grid_search.html#gridsearch-scoring).

  

## Grid Search

```
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
```

```Python
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
```



## Randomized Parameter Optimization

```python
print(__doc__)

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
```

  step1： 交叉验证（评价模型）

step2: 超参数选择，每一组参数：对应一次交叉验证

step 3: 集成学习

也可进行参数的调解

```Pytho
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores.mean()                             

```

```
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma='scale', kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1 = clf1.fit(X, y)
clf2 = clf2.fit(X, y)
clf3 = clf3.fit(X, y)
eclf = eclf.fit(X, y)
```

# sklearn.model_selection

### GridSearchCV

*class* `sklearn.model_selection.``GridSearchCV`(*estimator*, *param_grid*, ***, *scoring=None*, *n_jobs=None*, *iid='deprecated'*, *refit=True*, *cv=None*, *verbose=0*, *pre_dispatch='2\*n_jobs'*, *error_score=nan*, *return_train_score=False*)’

**estimator****estimator object.**

**param_grid****dict or list of dictionaries**

**scoring****str, callable, list/tuple or dict, default=None**

**cv****int, cross-validation generator or an iterable, default=None**

```python
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)


sorted(clf.cv_results_.keys())

```

