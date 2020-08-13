# 基础知识

## 参考资料

- [小猿取经](https://www.cnblogs.com/xiaoyuanqujing/p/11638188.html)
- [刘建平博客](https://www.cnblogs.com/pinard/category/894692.html)
- [李宏毅视频深度学习](https://www.bilibili.com/video/BV1JE411g7XF)
- [李沐动手学深度学习](https://space.bilibili.com/209599371/video)
- [OpenCV计算机视觉实战](https://www.bilibili.com/video/BV1ct411F7Te?p=2)

## 贝叶斯公式

> 可表相关性，不表因果，贝叶斯公式就是当已知结果，问导致这个结果的第i原因的可能性是多少，执果索因，用于计算逆概率

<img src="/note/tex/5ab3191b1822120550ec3267e66ddfb2.svg?invert_in_darkmode&sanitize=true" align=middle width=301.83224279999996pt height=24.65753399999998pt/>
> 前面为条件概率公式，后面为普通贝叶斯公式，。注意<img src="/note/tex/8b68470c60e215f6eef72e9d425c7b5c.svg?invert_in_darkmode&sanitize=true" align=middle width=181.13036864999998pt height=24.65753399999998pt/>的含义是不同的,<img src="/note/tex/9709ee9bd874ae9cffe07a2cded2bfc8.svg?invert_in_darkmode&sanitize=true" align=middle width=69.50910614999998pt height=24.65753399999998pt/>的样本是整体的，是全局的；而<img src="/note/tex/b0978fe53a31e379a438dd17684e71da.svg?invert_in_darkmode&sanitize=true" align=middle width=55.81063124999999pt height=24.65753399999998pt/>的样本是<img src="/note/tex/a900fc42abad5f1a1acd2dcbe8475f20.svg?invert_in_darkmode&sanitize=true" align=middle width=37.95100319999999pt height=24.65753399999998pt/>中的，<img src="/note/tex/58c27d7d54c494a3d7e91f398e0e8059.svg?invert_in_darkmode&sanitize=true" align=middle width=55.81063124999999pt height=24.65753399999998pt/>的样本是<img src="/note/tex/f540bd123ed34dd8062b411239b91955.svg?invert_in_darkmode&sanitize=true" align=middle width=38.91560969999999pt height=24.65753399999998pt/>中的，都是局部的

全概率公式
<img src="/note/tex/db069fca3428228a07c41f6ada1001e5.svg?invert_in_darkmode&sanitize=true" align=middle width=339.73373609999993pt height=26.438629799999987pt/>
>全概率公式的意义在于，当某一事件的概率难以求得时，可转化为在一系列条件下发生概率的和。

结合上面两个公式可得：
<img src="/note/tex/38349550cb72ce19eee30068280317f4.svg?invert_in_darkmode&sanitize=true" align=middle width=207.35193929999997pt height=33.20539859999999pt/>

## 独热编码 one hot encoding

独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

例如：

自然状态码为：000,001,010,011,100,101
独热编码为：000001,000010,000100,001000,010000,100000

## Soft-Max

将logist的值转化为概率，设<img src="/note/tex/65ed4b231dcf18a70bae40e50d48c9c0.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/>={1,2,3}，则Soft-Max为<img src="/note/tex/199d9802933f4f2e33708431193d6399.svg?invert_in_darkmode&sanitize=true" align=middle width=67.22871539999998pt height=29.943718200000013pt/>={0.09,0.244,0.665}

## 交叉熵 Cross Entropy

> 熵：混乱程度，熵越小越不混乱，熵为0表示一件确定的事情，表示两个值是完全相同的，熵越大表示两个值差距越大

用于计算两个值之间的距离，然后可以通过这个距离优化模型，即cross entropy充当loss function

<img src="/note/tex/ace39f01c85a900c10bdab09e1cc6cb8.svg?invert_in_darkmode&sanitize=true" align=middle width=238.80891704999993pt height=24.65753399999998pt/>

## 正向传播、反向传播

正向传播直接求当前参数(w、b)下函数的的值，反向传播求的是梯度，梯度，即斜率，即偏微分的值
<https://www.bilibili.com/video/BV1dW41187vW>
![图 1](images/Basic_2020-08-11_10-59-50.jpg)  

## 感知机

感知机接受多个输入信号，输出一个信号(多元一次函数)
<img src="/note/tex/a22f2c2ec25931cef440b44c4897beb9.svg?invert_in_darkmode&sanitize=true" align=middle width=40.02286529999999pt height=14.15524440000002pt/>为节点/神经元
![图 2](images/Basic_2020-08-12_01-38-10.jpg)
感知机可以模拟常用的逻辑电路，其中参数<img src="/note/tex/cd583e649d66b0a5efdbd7212a231f75.svg?invert_in_darkmode&sanitize=true" align=middle width=44.769907049999986pt height=14.15524440000002pt/>需要人手工调整，让机器根据需求调整<img src="/note/tex/cd583e649d66b0a5efdbd7212a231f75.svg?invert_in_darkmode&sanitize=true" align=middle width=44.769907049999986pt height=14.15524440000002pt/>的过程称为机器学习

## 似然函数

![图 18](images/Basic.tex_2020-08-13_03-21-56.png)  

## 机器学习算法优化

<https://blog.csdn.net/u012328159/article/details/80311892>
大致两类：
- 训练速度优化/收敛优化
> 优化的是学习率
- 过拟合/欠拟合优化
> 优化的是模型或者参数

## dropout

用于解决过拟合问题，主要思想是去掉一些神经元，相当于抽样训练，一种说法是减少多层神经元的耦合性，让每层神经元更好的抽象
![图 1](images/Basic.tex_2020-08-14_08-19-23.jpg)  
![图 2](images/Basic.tex_2020-08-14_08-29-22.jpg)  
![图 3](images/Basic.tex_2020-08-14_08-29-35.jpg)  
![图 4](images/Basic.tex_2020-08-14_08-39-51.jpg)  
