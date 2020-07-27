---
title: Python Basics
date: 2019-05-28 10:18:04
tags: Python
categories: [python]
---

# 重新学习

开始很乱的学习Python，现在想系统学习基础，真正了解pythonic,

<!-- more -->

# zip() function

a. function: **zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

b. zip([iterable, ...])

c. return : 返回元组列表。在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。

```python
a = [1,2,3]
b = [4,5,6]
zipped = zip(a,b) # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
zip(*zipped)  # 与 zip 相反，可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```



# enumerate() function

a. 将可遍历的数据对象(列表、元组或者字符串) 组合为一个索引序列，同时给出数据和数据下标。主要用于for循环

b. enumerate(sequence, [start=0])

c. return : enumerate(枚举) 对象。

d eg:

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
```

# Matplotlib bar() 和barh()

a. 绘制直方图和条形图，主要用于查看各个分组的数量分布

b. atplotlib.pyplot.bar(left, height, width=0.8, bottom=None, hold=None, data=None, **kwargs)

c.参数

| 参数      | 接收值 | 说明                 | 默认值 |
| --------- | ------ | -------------------- | ------ |
| left      | array  | x轴                  | 无     |
| height    | arrat  | 柱状图的高度         | 无     |
| alpha     | 数值   | 颜色透明度           | 1      |
| width     | 数值   | 宽度                 | 0.8    |
| color     | string | 填充颜色             | 随机色 |
| label     | string | 每个图像的代表的含义 | 无     |
| linewidth | 数值   | 线的刻度             | 1      |

d 例子

Demo 1: 基本骨架

```python
import pandas as pd
import matplotlib.pyplot as plt
 
#读取数据
datafile = u'D:\\pythondata\\learn\\matplotlib.xlsx'
data = pd.read_excel(datafile)

# 画布
plt.figure(figsize = (10, 5))
plt.title('Example of Histogram', fontsize = 20)
plt.xlabel(u'x-year', fontsize = 14)
plt.ylabel(u'y-income',fontsize = 14)

# 多个柱状图的分离距离
width_val = 0.4

plt.bar(data['time'],data['manincome'],width = width_val)
plt.bar(data['time']+width_val, data['femaleincome'],width = width_vale)

plt.legend(loc = 2)
plt.show()
```

Demo 2： 显示直方图的数值

```python
rect1 = plt.bar(data['time'],data['manincome'],width = width_val)
rect2 = plt.bar(data['time']+width_val, data['femaleincome'],width = width_vale)
python

# 添加数据标签

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, height, ha='center', va='bottom')
        rect.set_edgecolor('white')
```

Demo 3: 直方图堆叠显示（bottom 参数调节)

```python
plt.bar(data['time'],data['manincome'],width = width_val)
plt.bar(data['time'], data['femaleincome'], bottom = data['manincome'], width = width_vale)
python
```

## Case 2

![img](https://mmbiz.qpic.cn/mmbiz_jpg/giaycic3UNwo2WoxAYJvF4SvWfEN3q48eyKaHyekn7I8MwBllWzMuZDU28Tm5lygT1eDmKSYNxcAMpYsibWg6Cp6w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_jpg/giaycic3UNwo2WoxAYJvF4SvWfEN3q48eyQjcicSRGtBlXu4G2EkuPZjukH5ryo6g0PHW1zMlNl14hYhdOrJI0F6w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

实现细节：barh的参数 left设置：The x coordinates of the left sides of the bars (default: 0).

matplotlib.pyplot.barh(*y*, *width*, *height=0.8*, *left=None*, ,align='center', kwargs)

## DataFrame.interpolate

插值法，填充NaN

```
DataFrame.interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None, **kwargs)[source]
```

参数

```
method: str,default"linear'
'linear'
'time'
'index'
'nearest','zero','slinear',‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’

```

例子

```

```

# subplots_adjust(wspace, hspace)

function：调整子图间距

### DateTime

#### timedelta

时间和日期的计算

表示日期差

计算规则

![img](https://mmbiz.qpic.cn/mmbiz_png/e4kxNicDVcCHBKOC7JNA3oug3oeCoLOP6LbIBq1R1hSrmKQ3R8LVEaf0XWxXeFtibY2iaqH2O7N9nMM9GNlu0sLYA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```

from datetime import timedelta
td = timedelta(days=92) # days hours  minutes
print(d1 + td)
```

#### datetime.strptime()

字符串转换为日期和时间类型

```python
from datetime import datetime
cday = datetime.strptime('2017-8-1 18:20:20', '%Y-%m-%d %H:%M:%S')
```

## datetime.strftime()

datatime转化为字符串

```
from datetime import datetime
now = datetime.now()
print(now.strftime('%a, %b %d %H:%M'))
```

## DataFrame.rolling 窗口函数

```python
DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
```

参数说明：

window:时间窗的大小,数值int,即向前几个数据(可以理解将最近的几个值进行group by)
min_periods:最少需要有值的观测点的数量,对于int类型，默认与window相等
center:把窗口的标签设置为居中,布尔型,默认False
win_type: 窗口的类型,截取窗的各种函数。字符串类型，默认为None
on: 可选参数,对于dataframe而言，指定要计算滚动窗口的列,值为列名
closed：定义区间的开闭，支持int类型的window,对于offset类型默认是左开右闭的即默认为right,可以根据情况指定为left、both等
axis：方向（轴）,一般都是0

常用聚合函数：

mean() 求平均
count() 非空观测值数量
sum() 值的总和
median() 值的算术中值
min() 最小值
max() 最大
std() 贝塞尔修正样本标准差
var() 无偏方差
skew() 样品偏斜度（三阶矩）
kurt() 样品峰度（四阶矩）
quantile() 样本分位数（百分位上的值）
cov() 无偏协方差（二元）
corr() 相关（二进制）

------------------------------------------------
注意：设置的窗口window=3，也就是3个数取一个均值。index 0,1 为NaN