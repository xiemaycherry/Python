---
title: 数据分析
date: 2019-05-22 14:54:12
tags: [python, 数据分析]
categories: [数据分析]
---
<!-- more -->

# Pandas 做数据分析

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html?highlight=read_csv

### Step 1: 读取文件

- csv/txt
   1. **names**: columns，当names没被赋值时，header会变成0，即选取数据文件的第一行作为列名。当 names 被赋值，header 没被赋值时，那么header会变成None。如果都赋值，就会实现两个参数的组合功能。header = 0:是第一行是名字
   2. **sep**=‘\t’
   3. **header**=None:指定列名，数据开始行数。默认0行;None = 无标题
   4. **index_col** :None； 指定列作为行索引 数值。 False表示无索引
   5. usecols ; 如果列有很多，而我们不想要全部的列、而是只要指定的列就可以使用这个参数。
   6. **prefix** .prefix 参数，当导入的数据没有 header 时，设置此参数会自动加一个前缀。

   https://www.jianshu.com/p/42f1d2909bb6

### Step 2: 缺省值处理

#### 1. drop

`DataFrame.drop`**(***self***,** ***labels**=None***,** ***axis**=0***,** *index=None***,** *columns=None***,** *level=None***,** *inplace=False***,** *errors='raise'***)**

Parameters

- **labels** single label or list-like

  Index or column labels to drop.

- **axis** {0 or ‘index’, 1 or ‘columns’}, default 0

  Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

- **index** single label or list-like

  Alternative to specifying axis (`labels, axis=0` is equivalent to `index=labels`).*New in version 0.21.0.*

- **columns** single label or list-like

  Alternative to specifying axis (`labels, axis=1` is equivalent to `columns=labels`).*New in version 0.21.0.*

- **level **int or level name, optional

  For MultiIndex, level from which the labels will be removed.

- **inplace** bool, default False

  If True, do operation inplace and return None.

- **errors**{‘ignore’, ‘raise’}, default ‘raise’

  If ‘ignore’, suppress error and only existing labels are dropped.

#### 2. dropna

`DataFrame.dropna`**(***self***,** ***axis**=0***,** ***how**='any'***,** *thresh=None***,** ***subset**=None***,** ***inplace**=False***)**4

####  3. isna()

`DataFrame.isna`**(***self***)** 

2. 填充

#### 4. fillna()

DataFrame.fillna(**value**=None, **method**=None, **axis**=None, **inplace**=False, limit=None, downcast=None, **kwargs)

填充空值，values可以是字典 **values** = {'A': 0, 'B': 1, 'C': 2, 'D': 3} 

drop使用

1. **dropna**删除缺省值 df = df.drop(some labels) df = df.drop(df[<some boolean condition>].index)

   ​	axis: 0(index)

   ​	how: any all

   ​	subset: label

   ​	inplace : bool

2. **drop**删除 drop(labels(index, column labels), axis=0(行), level=None, inplace=False, errors='raise')

   ​	axis: label

   ​	index, columns 

3. **fillna **填充

   ​	fillna(value, method, limit)

### Step 3: 选取字段

1. 列
   1. df[labels]
2. 行
   1. df.loc[index_label,]
   2. df.iloc[整数值,]

列[""] 也可以传入条件语句

### 描述性统计函数

#### sum().mean().count() 

​		axis:1;按列求和，水平线 。往右看；所有列计算

​		axis:0; 按行求和, 垂直线；把字段的所有的所有行和。往下按 ；所有行计算

#### .describle() .transpose()

### 功能性函数

#### groupby[].

#### 多个DataFame归并

​    ### pd.merge(left, right, how='inner交集/Outer并集（存在不重合的key是', on='key'可以是一个列表[])

#### .apply()方法 传入函数名：然后每一个元素都背计算

​    lambda x : x*x

#### 排序

 .sort_values()整个表格都这么排列

  DataFrame.sort_values(by=[lables],axis=[0,1], ascending:, kind:排序算法，)

### 日期类型: 日期等数值处理

#### str列转换成日期类型 pd.to_datetime

注意非日期类型的特殊处理

```
df['日期时间'] = pd.to_datetime(df['日期时间'],format='%Y/%m/%d %H:%M:%S')
 
#获取 日期数据 的年、月、日、时、分
df['年'] = df['日期时间'].dt.year
df['月'] = df['日期时间'].dt.month
df['日'] = df['日期时间'].dt.day
df['时'] = df['日期时间'].dt.hour
df['分'] = df['日期时间'].dt.minute
```



![img](https://img-blog.csdnimg.cn/20190403095153639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MTg0Mzg4,size_16,color_FFFFFF,t_70)

#### 指定类型

​    method1 df1['year_month'] = df1['date'].apply(lambda x : x.strftime('%Y-%m'))

​    method2 df1['period'] = df1['date'].dt.**to_period**('M') 参数 M 表示月份，Q 表示季度，A 表示年度，D 表示按天

#### strp/ftime 

字符串和日期的转换

strftime: time->str

strptime: str->time
### datetime.timedelta
表示时间间隔，两个时间点之间的长度，主要用于时间计算,如时间序列预测的时候，需要外推，可能涉及到时间的计算
```python
timedelta(weeks=0, days=0, hours=0, minutes=0,  seconds=0, milliseconds=0, microseconds=0, )		 #依次为 "周" "天", "时","分","秒","毫秒","微秒"
```

#### datetime模块

| 类型     | 说明                                 |
| :------- | :----------------------------------- |
| date     | 以公历形式存储日历日期（年、月、日） |
| time     | 将时间存储为时、分、秒、毫秒         |
| datetime | 存储日期和时间                       |

**1）python标准库函数**

日期转换成字符串：利用str 或strftime

字符串转换成日期：datetime.strptime

```Python
stamp = datetime(2017,6,27)
str(stamp)
 '2017-06-27 00:00:00'
stamp.strftime('%y-%m-%d')#%Y是4位年，%y是2位年
 '17-06-27'
#对多个时间进行解析成字符串
date = ['2017-6-26','2017-6-27']
datetime2 = [datetime.strptime(x,'%Y-%m-%d') for x in date]
datetime2
[datetime.datetime(2017, 6, 26, 0, 0), datetime.datetime(2017, 6, 27, 0, 0)]
```

**3）pandas处理成组日期**

pandas通常用于处理成组日期，不管这些日期是DataFrame的轴索引还是列，to_datetime方法可以解析多种不同的日期表示形式。

**datetime 格式定义**

| 代码 | 说明                               |
| :--- | :--------------------------------- |
| %Y   | 4位数的年                          |
| %y   | 2位数的年                          |
| %m   | 2位数的月[01,12]                   |
| %d   | 2位数的日[01，31]                  |
| %H   | 时（24小时制）[00,23]              |
| %l   | 时（12小时制）[01,12]              |
| %M   | 2位数的分[00,59]                   |
| %S   | 秒[00,61]有闰秒的存在              |
| %w   | 用整数表示的星期几[0（星期天），6] |
| %F   | %Y-%m-%d简写形式例如，2017-06-27   |
| %D   | %m/%d/%y简写形式                   |

#### 字符串转换成datetime格式: strptime

datetime.strptime(str, '%Y/%m/%d').date()

#### datetime变回string格式: strftime

```
df = pd.DataFrame({"y": [1, 2, 3]},
...                   index=pd.to_datetime(["2000-03-31 00:00:00",
...                                         "2000-05-31 00:00:00",
...                                         "2000-08-31 00:00:00"]))
>>> df.index.to_period("M")
PeriodIndex(['2000-03', '2000-05', '2000-08'],
            dtype='period[M]', freq='M')
```

#### 1--pd.Period()参数：一个时间戳生成器

![img](https://upload-images.jianshu.io/upload_images/5798142-c03611e314662be4.png?imageMogr2/auto-orient/strip|imageView2/2/w/966/format/webp)

### Step 4: 数据透析表 pivot_table

#### .pivot_table 数据透析表 分类汇总的统计数据

​     (data,values= column to aggregate optional, index = grouper, columns=grouper, aggfunc:np.sum  )

​     table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean, 'E': np.mean})

### .groupby()

![img](https://upload-images.jianshu.io/upload_images/2862169-51af7d4ae64c2f78.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

由于通过`groupby()`函数分组得到的是一个`DataFrameGroupBy`对象，而通过对这个对象调用`get_group()`，返回的则是一个·DataFrame·对象，所以可以将`DataFrameGroupBy`对象理解为是多个`DataFrame`组成的。

```python
grouped = df.groupby('Gender')
grouped_muti = df.groupby(['Gender', 'Age'])

```

```python
print(grouped.get_group('Female'))
print(grouped_muti.get_group(('Female', 17)))

    Name  Gender  Age  Score
2   Cidy  Female   18     93
4  Ellen  Female   17     96
7   Hebe  Female   22     98
    Name  Gender  Age  Score
4  Ellen  Female   17     96
```

```python
print(grouped.count())
print(grouped.max()[['Age', 'Score']])
print(grouped.mean()[['Age', 'Score']])
        Name  Age  Score
Gender                  
Female     3    3      3
Male       5    5      5
        Age  Score
Gender            
Female   22     98
Male     21    100
         Age      Score
Gender                 
Female  19.0  95.666667
Male    19.6  89.000000
```

如果其中的函数无法满足你的需求，你也可以选择使用聚合函数`aggregate`，传递`numpy`或者自定义的函数，前提是返回一个聚合值

```python
def getSum(data):
    total = 0
    for d in data:
        total+=d
    return total


print(grouped.aggregate(np.median))
print(grouped.aggregate({'Age':np.median, 'Score':np.sum}))
print(grouped.aggregate({'Age':getSum}))
```

迭代

```python
grouped = df.groupby('A')
for name, group in grouped:
	    print(name)
  		print(group)
        
bar
     A      B         C         D
1  bar    one  0.254161  1.511763
3  bar  three  0.215897 -0.990582
5  bar    two -0.077118  1.211526
foo
     A      B         C         D
0  foo    one -0.575247  1.346061
2  foo    two -1.143704  1.627081
4  foo    two  1.193555 -0.441652
6  foo    one -0.408530  0.268520
7  foo  three -0.862495  0.024580
```

可视化

对组内的数据绘制概率密度分布：

```python
grouped['Age'].plot(kind='kde', legend=True)
plt.show()
```

计算不同组的某一列的值

```python
data.groupby('race')['age'].mean()
要求被不同种族内被击毙人员年龄的均值:
```

对不同取值的计数: `.value_counts()`

```python
data.groupby('race')['signs_of_mental_illness'].value_counts()
求不同种族内, 是否有精神异常迹象的分别有多少人
```

```python
data.groupby('race')['signs_of_mental_illness'].value_counts().unstack()
组内操作的结果不是单个值, 是一个序列, 我们可以用.unstack()将它展开，得到DateFrame
```

```python
data.groupby('race')['flee'].value_counts().unstack().plot(kind='bar', figsize=(20, 4))
```

![img](https://upload-images.jianshu.io/upload_images/2862169-e149898f78d62a81.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

这里有一个之前介绍的`.unstack`操作, 这会让你得到一个`DateFrame`, 然后调用条形图, pandas就会遍历每一个组(`unstack`后为每一行), 然后作各组的条形图

### 按不同逃逸类型分组, 组内的年龄分布是如何的?

```python
data.groupby('flee')['age'].plot(kind='kde', legend=True, figsize=(20, 5))
```

![img](https://upload-images.jianshu.io/upload_images/2862169-1a6135108e73ebc2.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

这里`data.groupby('flee')['age']`是一个`SeriesGroupby`对象, 顾名思义, 就是每一个组都有一个`Series`. 因为划分了不同逃逸类型的组, 每一组包含了组内的年龄数据, 所以直接`plot`相当于遍历了每一个逃逸类型, 然后分别画分布图.


### Step 5: 写入表格

to_csv（path_or_buf，sep，header: **bool** or list of str : default：true， index: **bool**, default true）

## 字符串处理

### 多个字符串分割

Python中的spilt方法只能通过指定的某个字符分割字符串，如果需要指定多个字符，需要用到re模块里的split方法。

```
>>> import re

>>> a = "Hello world!How are you?My friend.Tom"

>>> re.split(" |!|\?|\.", a)

['Hello', 'world', 'How', 'are', 'you', 'My', 'friend', 'Tom']
```

### 去掉多余空格

1. filter 

aStr_splited = aStr.split(' ') 

print(filter(lambda x : x, aStr_splited))

list(filter(None,s.split(',')))

2. 列表

   [x for x in s.split(',') if x]

## 正则表达式

https://docs.python.org/zh-cn/3/library/re.html

正则表达式：一种特殊的字符串，用于查找某种形式的字符串，满足某种条件的格式。

用于查找，匹配

re模块

re.

pattern

​	[a-z] [abc]

​    [^ab]

```
\d
```

匹配任何十进制数字；这等价于类 `[0-9]`。

```
\D
```

匹配任何非数字字符；这等价于类 `[^0-9]`。

```
\s
```

匹配任何空白字符；这等价于类 `[ \t\n\r\f\v]`。

```
\S
```

匹配任何非空白字符；这相当于类 `[^ \t\n\r\f\v]`。

```
\w
```

匹配任何字母与数字字符；这相当于类 `[a-zA-Z0-9_]`。

```
\W
```

匹配任何非字母与数字字符；这相当于类 `[^a-zA-Z0-9_]`。

`ca*t` 将匹配 `'ct'` (0个 `'a'` 字符)，`'cat'` (1个 `'a'` )， `'caaat'` (3个 `'a'` 字符)

另一个重复的元字符是 `+`，它匹配一次或多次。 要特别注意 `*` 和 `+` 之间的区别；`*` 匹配 *零次* 或更多次，因此重复的任何东西都可能根本不存在，而 `+` 至少需要 *一次*。 使用类似的例子，`ca+t` 将匹配 `'cat'` (1 个 `'a'`)，`'caaat'` (3 个 `'a'`)，但不会匹配 `'ct'`。

最复杂的重复限定符是 `{m,n}`，其中 *m* 和 *n* 是十进制整数。 这个限定符意味着必须至少重复 *m* 次，最多重复 *n* 次。 例如，`a/{1,3}b` 将匹配 `'a/b'` ，`'a//b'` 和 `'a///b'` 。 它不匹配没有斜线的 `'ab'`，或者有四个的 `'a////b'`。

`^`表示行的开头，`^\d`表示必须以数字开头。

`$`表示行的结束，`\d$`表示必须以数字结束。

| 方法 / 属性  | 目的                                                         |
| :----------- | :----------------------------------------------------------- |
| `match()`    | 确定正则是否从字符串的开头匹配。                             |
| `search()`   | 扫描字符串，查找此正则匹配的任何位置。                       |
| `findall()`  | 找到正则匹配的所有子字符串，并将它们作为列表返回。           |
| `finditer()` | 找到正则匹配的所有子字符串，并将它们返回为一个 [iterator](https://docs.python.org/zh-cn/3/glossary.html#term-iterator)。 |

匹配对象返回值的函数

方法 / 属性

目的

```
group()
```

返回正则匹配的字符串

```
start()
```

返回匹配的开始位置

```
end()
```

返回匹配的结束位置

```
span()
```

返回包含匹配 (start, end) 位置的元组

```
>>> m.group()
'tempo'
>>> m.start(), m.end()
(0, 5)
>>> m.span()
(0, 5)
```

- `re.``search`(*pattern*, *string*, *flags=0*)[¶](https://docs.python.org/zh-cn/3/library/re.html#re.search)

  扫描整个 *字符串* 找到匹配样式的第一个位置，并返回一个相应的 [匹配对象](https://docs.python.org/zh-cn/3/library/re.html#match-objects)。如果没有匹配，就返回一个 `None` ； 注意这和找到一个零长度匹配是不同的。

- `re.``match`(*pattern*, *string*, *flags=0*)

  如果 *string* 开始的0或者多个字符匹配到了正则表达式样式，就返回一个相应的 [匹配对象](https://docs.python.org/zh-cn/3/library/re.html#match-objects) 。 如果没有匹配，就返回 `None` ；注意它跟零长度匹配是不同的。



## pandas

```
df.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='best')
```

**data** : DataFrame

**x** : label or position, default None

**y** : label or position, default None

> Allows plotting of one column versus another

**kind** : str

> - ‘line’ : line plot (default)
> - ‘bar’ : vertical bar plot
> - ‘barh’ : horizontal bar plot
> - ‘hist’ : histogram
> - ‘box’ : boxplot
> - ‘kde’ : Kernel Density Estimation plot
> - ‘density’ : same as ‘kde’
> - ‘area’ : area plot
> - ‘pie’ : pie plot
> - ‘scatter’ : scatter plot
> - ‘hexbin’ : hexbin plot

**ax** : matplotlib axes object, default None



### 时间序列

```python
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

fig, ax = plt.subplots()

months = mdates.MonthLocator()

dateFmt = mdates.DateFormatter("%m/%d/%y")

ax.xaxis.set_major_formatter(dateFmt)
ax.xaxis.set_minor_locator(months)
ax.tick_params(axis="both", direction="out", labelsize=10)

date1 = datetime.date(2005, 8, 8)
date2 = datetime.date(2015, 6, 6)
delta = datetime.timedelta(days=5)
dates = mdates.drange(date1, date2, delta)

y = np.random.normal(100, 15, len(dates))

ax.plot_date(dates, y, "#FF8800", alpha=0.7)

fig.autofmt_xdate()

plt.show()
```

#### plot()

```
DataFrame.plot(self, *args, **kwargs)
kindstr
The kind of plot to produce:
‘line’ : line plot (default)
‘bar’ : vertical bar plot
‘barh’ : horizontal bar plot
‘hist’ : histogram
‘box’ : boxplot
‘kde’ : Kernel Density Estimation plot
‘density’ : same as ‘kde’
‘area’ : area plot
‘pie’ : pie plot
‘scatter’ : scatter plot
‘hexbin’ : hexbin plot.

figsizea tuple (width, height) in inches
x :label
y:label
xlim:
xticks:
title

pandas.DataFrame.plot.bar
pandas.DataFrame.plot.barh
pandas.DataFrame.plot.box
pandas.DataFrame.plot.density
pandas.DataFrame.plot.hexbin
pandas.DataFrame.plot.hist
pandas.DataFrame.plot.kde
pandas.DataFrame.plot.line
pandas.DataFrame.plot.pie
pandas.DataFrame.plot.scatter
pandas.DataFrame.boxplot
pandas.DataFrame.hist
```





https://blog.csdn.net/fengbingchun/article/details/81035861?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase

https://www.cnblogs.com/Summer-skr--blog/p/11705925.html
### 时间序列绘图
坐标轴xtick设置方法
plt.xticks(statiem,[datetime.strftime(x,'%Y-%m') for x in statiem])
## 注意事项

### list() dict()的拷贝

1、**b = a:** 赋值引用，a 和 b 都指向同一个对象。

![img](https://www.runoob.com/wp-content/uploads/2017/03/1489720931-7116-4AQC6.png)

**2、b = a.copy():** 浅拷贝, a 和 b 是一个独立的对象，但他们的子对象还是指向统一对象（是引用）。

![img](https://www.runoob.com/wp-content/uploads/2017/03/1489720930-6827-Vtk4m.png)

**b = copy.deepcopy(a):** 深度拷贝, a 和 b 完全拷贝了父对象及其子对象，两者是完全独立的。

![img](https://www.runoob.com/wp-content/uploads/2017/03/1489720930-5882-BO4qO.png)

## 访问

1. df[-1:] *#最后一行*
2. df[-3:-1] *#倒数第3行到倒数第1行（不包含最后1行即倒数第1行）

# Linux

## nohup

linux后台执行命令：&和nohup

用途：不挂断地运行命令。

语法：nohup Command [ Arg … ] [　& ]

例子： nohup sh example.sh &

`nohup` 命令可以使命令永久的执行下去，和终端没有关系，退出终端也不会影响程序的运行； 
`&` 是后台运行的意思，但当用户退出的时候，命令自动也跟着退出。 
**那么，把两个结合起来`nohup 命令 &`这样就能使命令永久的在后台执行**

`nohup 命令 > output.log 2>&1 &`让命令在后台执行。

其中 0、1、2分别代表如下含义： 
0 – stdin (standard input) 
1 – stdout (standard output) 
2 – stderr (standard error)

`nohup`+最后面的`&`是让命令在后台执行

`>output.log` 是将信息输出到output.log日志中

`2>&1`是将标准错误信息转变成标准输出，这样就可以将错误信息输出到output.log 日志里面来。

& 后台执行
\>  输出到
不过联合使用也有其他意思，比如nohup输出重定向上的应用
例子：nohup abc.sh > nohup.log 2>&1 &
其中2>&1  指将[STDERR](https://www.baidu.com/s?wd=STDERR&tn=SE_PcZhidaonwhc_ngpagmjz&rsv_dl=gh_pc_zhidao)重定向到前面标准输出定向到的同名文件中，即&1就是nohup.log

## ps -ef|grep python

ps命令将某个进程显示出来

grep命令是查找

中间的|是管道命令 是指ps命令与grep同时执行

PS是LINUX下最常用的也是非常强大的进程查看命令

grep命令 是查找， 是一种强大的文本搜索工具，它能 [使用正则表达式 ](https://www.baidu.com/s?wd=使用正则表达式&tn=44039180_cpr&fenlei=mv6quAkxTZn0IZRqIHckPjm4nH00T1d9uWD3PhP9n1b4m1nduAcz0ZwV5Hcvrjm3rH6sPfKWUMw85HfYnjn4nH6sgvPsT6KdThsqpZwYTjCEQLGCpyw9Uz4Bmy-bIi4WUvYETgN-TLwGUv3EPjfvrHnzPWT3)搜索文本，并把匹 配的行打印出来。

grep全称是Global Regular Expression Print，表示全局正则表达式版本，它的使用权限是所有用户。

以下这条命令是检查 java 进程是否存在：ps -ef |grep java

字段含义如下：
UID    PID    PPID    C   STIME   TTY    TIME     CMD

zzw   14124  13991   0   00:38   pts/0   00:00:00  grep --color=auto dae

 

**UID   ：程序被该 UID 所拥有**

**PID   ：就是这个程序的 ID** 

**PPID  ：则是其上级父程序的ID**

**C     ：CPU使用的资源百分比**

**STIME ：系统启动时间**

**TTY   ：登入者的终端机位置**

**TIME  ：使用掉的CPU时间。**

**CMD  ：所下达的是什么指令**