# Regression

- [Regression](#regression)
  - [线性回归步骤](#线性回归步骤)
    - [1 线性回归公式](#1-线性回归公式)
    - [2 损失函数](#2-损失函数)
    - [3 梯度下降](#3-梯度下降)
    - [4 过拟合](#4-过拟合)
    - [5 正则化](#5-正则化)

机器学习：让机器找出一个解决问题的函数
比如这样的函数：

- 输入昨天的天气，输出今天的天气
- 输入路况，输出油门和方向盘的大小

监督学习(Supervised): 需要提供已标签化的数据集(即正确的输入、输出都需要准备好)
强化学习(Reinforcement): 使用某种奖励机制，让机器自己找出理想的答案?
想法： 训练出完股票游戏的程序

## 线性回归步骤

### 1 线性回归公式

<img src="/note/tex/41e36ad7f316c3c7b0c4405eedec721c.svg?invert_in_darkmode&sanitize=true" align=middle width=93.51777764999999pt height=31.02729300000001pt/>

> 上标：某一对象
> 下标：某一对象的一个属性

目标是找出w,b参数的值使得输入某一x时，得到的y最准确

### 2 损失函数

通过已知的数据集，也就是y减去，可得损失函数：

<img src="/note/tex/d2e83a4e5a5e7004ac2d79df3720f4fc.svg?invert_in_darkmode&sanitize=true" align=middle width=318.67460789999996pt height=31.36100879999999pt/>

当存在w，b使得该损失函数最小时，则w和b为(局部)最优解：

<img src="/note/tex/cf964d00346249677d99d8a3f01448c8.svg?invert_in_darkmode&sanitize=true" align=middle width=200.16300314999998pt height=24.65753399999998pt/>

### 3 梯度下降

(不仅仅适用于线性回归，只要函数<img src="/note/tex/e175ea32fe673804e9313fe7b9119ebd.svg?invert_in_darkmode&sanitize=true" align=middle width=33.790089299999984pt height=24.65753399999998pt/>是可微分的都可以用)

> 导数为瞬时变化率：
> <img src="/note/tex/3b14b87ed505618b8b3e8e8d24c99e1b.svg?invert_in_darkmode&sanitize=true" align=middle width=462.71267955pt height=33.20539859999999pt/>
> 微分：
> <img src="/note/tex/11115845770e08b5204019ae3ad724c2.svg?invert_in_darkmode&sanitize=true" align=middle width=153.41515529999998pt height=24.7161288pt/>

![图 1](images/P3_Regression_2020-07-22_07-14-20.png)  

(?为什么<img src="/note/tex/393655c6934375e41496acf410220123.svg?invert_in_darkmode&sanitize=true" align=middle width=36.18352154999999pt height=24.65753399999998pt/>图像是长这样的，猜测：为教学目的这里才使用这样的曲线，通过观察<img src="/note/tex/393655c6934375e41496acf410220123.svg?invert_in_darkmode&sanitize=true" align=middle width=36.18352154999999pt height=24.65753399999998pt/>方程式 的方程式可知，它的图像应该是一个类似二次函数的图像，也就是有全局最优解的图像，即局部最优解为全局最优解)

1. 不管参数b，先处理w，得<img src="/note/tex/56c3cc8e0139c40d7b13344c075e4d1a.svg?invert_in_darkmode&sanitize=true" align=middle width=143.46596879999998pt height=24.65753399999998pt/>
2. 随机选择一个<img src="/note/tex/dee3368366fc5ad01199b96a460aaf52.svg?invert_in_darkmode&sanitize=true" align=middle width=18.76339244999999pt height=26.76175259999998pt/>
3. 更新<img src="/note/tex/027a4bbbe5866d81528dd0e578d95066.svg?invert_in_darkmode&sanitize=true" align=middle width=167.99703029999998pt height=28.92634470000001pt/>，启动<img src="/note/tex/1d0496971a2775f4887d1df25cea4f7e.svg?invert_in_darkmode&sanitize=true" align=middle width=8.751954749999989pt height=14.15524440000002pt/>称作学习率

重复上述步骤3，直到<img src="/note/tex/393655c6934375e41496acf410220123.svg?invert_in_darkmode&sanitize=true" align=middle width=36.18352154999999pt height=24.65753399999998pt/>取得最小值
此时b也可以类似这样处理，通过偏微分来更新w和b：
![图 2](images/P3_Regression_2020-07-22_07-21-37.png)  

w和b组成的图像则如同等高线图：
![图 3](images/P3_Regression_2020-07-22_07-25-05.png)  

### 4 过拟合

![图 4](images/P3_Regression_2020-07-22_07-48-15.png)  
![图 5](images/P3_Regression_2020-07-22_08-01-17.jpg)  

### 5 正则化

回到第2步骤，调整Loss function为：
<img src="/note/tex/56ac17d36ea43a10fc495cd72a99830a.svg?invert_in_darkmode&sanitize=true" align=middle width=284.04775904999997pt height=26.76175259999998pt/>
该式期望<img src="/note/tex/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41940739999999pt height=14.15524440000002pt/>的取值越小越好，其原理是去除无效数据或者干扰数据的影响

![图 6](images/P3_Regression_2020-07-22_08-28-25.jpg)  
