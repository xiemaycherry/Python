---

title: python库
date: 2019-02-24 12:29:29
tags: Python
categories: [python]
---
开始接触Python是大二结束的时候，到现在都快两年了，其实一直并不是很细节的学习，只是希望能够跑个结果。不过呢？，以后肯定是会经常用Python，所以呢？我接下来会认真学习Python

<!-- more -->

## Python 高级用法总结

基本数据类型：整型、浮点型、布尔类型
### 容器： Containers
容器是一种把多个元素组织在一起的数据结构，容器中的元素可以逐个地迭代获取，可以用in, not in关键字判断元素是否包含在容器中。通常这类数据结构把所有的元素存储在内存中（也有一些特例，并不是所有的元素都放在内存，比如迭代器和生成器对象）在Python中，常见的容器对象有：
list, deque
set, frozensets
dict, defaultdict, OrderedDict, Counter
tuple, namedtuple
str
 ## list推导（list comprehensions)

官方解释：列表解析式是Python内置的非常**简单**却**强大**的可以用来创建list的生成式。

```la
对于一个列表，既要遍历索引又要遍历元素。
```

```Py
array = ['I', 'love', 'Python']
for i, element in enumerate(array):
    array[i] = '%d: %s' % (i, seq[i])
```

```Py
def getitem(index, element):
    return '%d: %s' % (index, element)

array = ['I', 'love', 'Python']
arrayIndex = [getitem(index, element) for index, element in enumerate(array)]
```

## 迭代器和生成器
### 可迭代对象：
凡是可以返回一个迭代器的对象都可称之为可迭代对象
例如：list    dic    str     set     tuple     range()     enumerate(枚举)     f=open()（文件句柄）
```Py
### 迭代器(iterator)
是一个带状态的对象，他能在你调用next()方法的时候返回容器中的下一个值，任何实现了__iter__和__next__()（python2中实现next()）方法的对象都是迭代器，__iter__返回迭代器自身，__next__返回容器中的下一个值，如果容器中没有更多元素了，则抛出StopIteration异常
### 生成器(generator)
生成器其实是一种特殊的迭代器，不过这种迭代器更加优雅。它不需要再像上面的类一样写__iter__()和__next__()方法了，只需要一个yiled关键字。 生成器一定是迭代器（反之不成立）
#列表生成式
lis = [x*x for x in range(10)]
# 受到内存限制，列表容量肯定是有限的
#生成器表达式
generator_ex = (x*x for x in range(10))
```
生成器： 不用创建完整的list，为节省大量的空间，在Python中，这种一边循环一边计算的机制，称为生成器：generator
Tuples:()
 字典：{：，}
 Sets: {,}
函数
类

## Python库----numpy
### What
NumPy=Numerical+Python
主要是提供了高性能多维数组这个对象，以及处理相关的方法
### How
1. 自定义一个（1D or MD)数组或者特殊的数组,一维，二维
2. 数组切片（也就是提取数组元素），注意 a[:,0]和a[:,0:1]是不同的喔
3. 关于数组属性的方法
4. 数组运算
5. 索引
    where 函数
    索引的布尔数组
6.  广播（Broadcasting）
    用于处理不同性状的 数组。 Broadcasting提供了一种矢量化数组操作的方法，使得循环发生在C而不是Python。标量乘以一个矢量的时候，用Boradcasting更快，因为 broadcasting在乘法期间移动较少的内存
7. array 和 matrix 选择哪个?
    [戳我](https://www.numpy.org.cn/user_guide/numpy_for_matlab_users.html)
8. 矢量化和广播、索引
在Python中循环数组或任何数据结构时，会涉及很多开销。 NumPy中的向量化操作将内部循环委托给高度优化的C和Fortran函数，从而实现更清晰，更快速的Python代码。
## stack|vstack|hstack

```python
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.stack((a, b))
array([[1, 2, 3],
       [2, 3, 4]])

% hstack
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])
a = np.array([[1],[2],[3]])
b = np.array([[2],[3],[4]])
np.hstack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
% vstack
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```



## mean

```
a = np.array([[1, 2], [3, 4]])
np.mean(a)

np.mean(a, axis=0)

np.mean(a, axis=1)
```

## reshape

reshape(x, y)，其中x表示转换后数组的行数，y表示转换后数组的列数。当x或者y为-1时，表示该元素随机分配，如reshape(2, -1)表示列数随机，行数为两行。

```
格式：np.reshape((x, y, z))

参数的含义：

x：表示生成的三维数组中二维数组的个数

y：表示单个二维数组中一维数组的个数

z：表示三维数组的列数
```

## numpy数组去掉冗余的维度-----squeeze()函数

import numpy as np a = [[[10, 2, 3]]] a = np.array(a) a_sque = np.squeeze(a) print(a) print(a_sque)

## Python库----pandas
记得学习pandas是在大三时候的美赛，花了一天多时间学习pandas，然后预处理数据，当时三个队友都是各自的家，是非常愉快的！！！
### what
Python Data Analysis Library
1. 三种数据结构
序列： Series 1D
数据帧： DataFrame 2D
面板： Panel >2D
2. 自定义创建
   1. 可以通过字段、数据、series、列表
   2. 列表传入的时候，主要行列，如果单个列表：列；如果是[[],[]]是按行[]
   3. 如果位置不对可转置
   4. 创建空 pd.DataFrame()
3. 选择区块
    a) Series
    []
    b) DataFrame
    列选择 ['colums的名字']
    行列选择：.loc[列名,行名]名称 .iloc[列索引,行索引]整数
4. array
   .value
5. 统计描述
    .descibe(include = 'all') .head() .tail()
    .select_dtype(include=[])
    .columns
    .dtype
6. 缺少数据
    1. 查看缺失值
       isnull() notnull() 也可以 做一些统计，sum, any,all 
    2. 清理缺失值
        dropna(axis=0)：axis = 0:index axis=1,columns
    3. 填充缺少指
        fillna() 标量替换
    4. 替换
    
7. 统计函数


8. Pandas 函数应用
表合理函数应用：pipe()
行或列函数应用：apply()
元素函数应用：applymap()
eg： pd.pipe(lambda x: x*100)

9. 类别变量向量化
非数值类型的处理方法

10. 时间序列生成 data_range
    1.  pandas.date_range("11:00", "21:30", freq="30min")
    2.  参数
    ```python
    Return a fixed frequency DatetimeIndex.
    ```

Parameters
startstr or datetime-like, optional
Left bound for generating dates.

endstr or datetime-like, optional
Right bound for generating dates.

periodsint, optional
Number of periods to generate.

freqstr or DateOffset, default ‘D’
Frequency strings can have multiples, e.g. ‘5H’. See here for a list of frequency aliases.

tzstr or tzinfo, optional
Time zone name for returning localized DatetimeIndex, for example ‘Asia/Hong_Kong’. By default, the resulting DatetimeIndex is timezone-naive.

normalizebool, default False
Normalize start/end dates to midnight before generating date range.

namestr, default None
Name of the resulting DatetimeIndex.

closed{None, ‘left’, ‘right’}, optional
Make the interval closed with respect to the given frequency to the ‘left’, ‘right’, or both sides (None, the default).

**kwargs
For compatibility. Has no effect on the result.

Returns
rngDatetimeIndex
```
11. DataFrame.stack
Parameters
levelint, str, list, default -1
Level(s) to stack from the column axis onto the index axis, defined as one index or label, or a list of indices or labels.

dropnabool, default True
Whether to drop rows in the resulting Frame/Series with missing values. Stacking a column level onto the index axis can create combinations of index and column values that are missing from the original dataframe. See Examples section.

Returns
DataFrame or Series
Stacked dataframe or series.
​```python
df_single_level_cols
     weight height
cat       0      1
dog       2      3
df_single_level_cols.stack()
cat  weight    0
     height    1
dog  weight    2
     height    
```
DataFrame.value_connts()返回序列，index=统计值，值：统计个数

## Matplotlib
matplotlib.pyplot as plt
1.  窗口：figure: 一个窗口，plt.figure(num=,figsize=(h,w))下面数据都属于当前的figure,有一定的顺序喔
2.  画图：plt.plot(x,y,color=,linewidth=,linestyle,label=)
3.  标注信息： plt.xlim((,)), plt.yxlim((,)),plt.xlabel(),plt.ylabel(),ticks:图像的小标，plt.xticks(),plt.yticks([值1，值2],[r'$值1\ 对应的文字$',r'值2的文字 \alpha])
4.  坐标轴：axis gac='get current axis'
ax = plt.gca() # 轴
\# 获取四个轴
ax.spines['right|left|top|'].set_color('none') 
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',-1))
5. 图例：legend: 
    a. plt.plot(,label=), plt.legend()
    b. l1, = plt.plot() plt.legend(handles=[l1,],labels=[,],loc='best|upper right|')
6. 注解 annotation 
a. 点的位置(x0，y0) plt.scatter(). plt.plot([x0,y0],[y0,0],'k--',lw=)
b . method 1:
plt.annotate(r'name',xy=(,)起始点，xycoords='data'//基于xy,xytext=(+30,30),textcoords='offseet points'//文本基于xy,arrowprops=dict(arrowstyle='->'箭头,connectionstyle='arc3,rad=.2')弧度)
7. Bar 柱状图
plt.bar(x,+|-y,facecolor="",edgecolor,)
|# ha horizontal alignment 对齐方式
for x,y in zip(x,y):
    plt.text(x+0,4,y+0.05,'%.2f'%y,ha='center',va='bottom')
8. 很多自动 subplot(总行，当前行的列，总的按最小分的第几个)
subplot(,,)
## index

reset_index:限于DataFrame

set_index

index

# scikit-learn

官方教程绝对是最好最棒的选择，有简单数学推导、直观立马就能上手的案例，还能提阅读英文的能力喔，实在是一举多得啊！！！！ [scikit-learn.org](https://scikit-learn.org/)

## regression

## Feature selection

### Method from sklearn.feature_selection import VarianceThreshold



### sklearn.feature_selection.SelectFromModel

*class* sklearn.feature_selection.SelectFromModel(*estimator*, , *threshold=None*, *prefit=False*, *norm_order=1*, *max_features=None*)

# 

# seaborn

seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=None, color=None, height=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)

- Parameters

  **x, y**strings or vectorsData or names of variables in `data`.**data**DataFrame, optionalDataFrame when `x` and `y` are variable names.**kind**{ “scatter” | “reg” | “resid” | “kde” | “hex” }, optionalKind of plot to draw.**stat_func**callable or None, optional*Deprecated***color**matplotlib color, optionalColor used for the plot elements.**height**numeric, optionalSize of the figure (it will be square).**ratio**numeric, optionalRatio of joint axes height to marginal axes height.**space**numeric, optionalSpace between the joint and marginal axes**dropna**bool, optionalIf True, remove observations that are missing from `x` and `y`.**{x, y}lim**two-tuples, optionalAxis limits to set before plotting.**{joint, marginal, annot}_kws**dicts, optionalAdditional keyword arguments for the plot components.**kwargs**key, value pairingsAdditional keyword arguments are passed to the function used to draw the plot on the joint Axes, superseding items in the `joint_kws` dictionary.

- Returns

  **grid**[`JointGrid`](https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid)[`JointGrid`](https://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid) object with the plot on it.

http://seaborn.pydata.org/generated/seaborn.JointGrid.html#seaborn.JointGrid

g = sns.jointplot(x="x", y="y", kind = 'reg' , space=0,color = 'g', data=df11,stat_func=sci.pearsonr)

sns.set()

sns.axes_style("darkgrid")

sns.set_context("paper")

https://blog.mazhangjing.com/2018/03/29/learn_seaborn/

https://blog.csdn.net/weiyudang11/article/details/51549672

```
#初始化类
g=sns.JointGrid(x='v_ma5',y='price_change',data=stock,space=0.5,ratio=5)

g=sns.JointGrid(x='v_ma5',y='price_change',data=stock,space=0.5,ratio=5)
g=g.plot_joint(plt.scatter,color='.3',edgecolor='r')
g=g.plot_marginals(sns.distplot,kde=False)


from scipy import stats
g=sns.JointGrid(x='v_ma5',y='price_change',data=stock,space=0.5,ratio=5)
g=g.plot_joint(plt.scatter,color='.3',edgecolor='r')
_=g.ax_marg_x.hist(stock.v_ma10,color='r',alpha=.6,bins=50)
_=g.ax_marg_y.hist(stock.low,color='y',orientation="horizontal",bins=20)
rquare=lambda a,b:stats.pearsonr(a,b)[0]**2
g=g.annotate(rquare,template='{stat}:{val:.2f}',stat='$R^2$',loc='upper left',fontsize=12)
```

## 颜色和风格设置

## 调色板

主要使用以下几个函数设置颜色：
color_palette() 能传入任何Matplotlib所有支持的颜色
color_palette() 不写参数则默认颜色

current_palette = sns.color_palette() sns.palplot(current_palette) plt.show()

set_palette() 设置所有图的颜色

sns.palplot(sns.color_palette("hls",8)) plt.show()

## 颜色的亮度及饱和度

l-光度 lightness
s-饱和 saturation

sns.palplot(sns.hls_palette(8,l=.7,s=.9)) plt.show()

## xkcd选取颜色

xkcd包含了一套众包努力的针对随机RGB色的命名。产生了954个可以随时通过xkcd_rgb字典中调用的命名颜色

plt.plot([0,1],[0,1],sns.xkcd_rgb['pale red'],lw = 3) #lw = 线宽度
plt.plot([0,1],[0,2],sns.xkcd_rgb['medium green'],lw = 3)
plt.plot([0,1],[0,3],sns.xkcd_rgb['denim blue'],lw = 3)
plt.show()

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190216223007343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ryb2tlX1pob3U=,size_16,color_FFFFFF,t_70#pic_center)

## 汇总

http://seaborn.pydata.org/api.html#

https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py

https://xkcd.com/color/rgb/