# P20 RNN(Recurrent Neural Network，循环神经网络)

![图 4](images/P20_RNN_2020-08-12_02-03-19.png)  

## 语言模型

![图 6](images/P20_RNN_2020-08-12_03-27-19.png)  

## 梯度消失(Gradient Vanishing)和梯度爆炸(Gradient Exploding)

![图 7](images/P20_RNN_2020-08-12_03-37-49.png)  

当序列很长的时候，不凑巧所有微分值很小或很大的时候，会出现梯度消失(值为0)和梯度爆炸问题(值为无穷大)，可用gradient clipping解决

## 长短期记忆LSTM(Long short term memory)

是其中一种比较常用RNN的变种，让程序拥有记忆能力
可解决梯度消失和梯度爆炸问题
![图 8](images/P20_RNN_2020-08-12_03-58-23.png)  
输入门、遗忘门、输出门、候选细胞、记忆细胞
![图 15](images/P20_RNN_2020-08-12_10-58-52.jpg)  

其他简化版：
![图 10](images/P20_RNN_2020-08-12_04-00-52.png)  
![图 11](images/P20_RNN_2020-08-12_04-01-01.png)  
![图 12](images/P20_RNN_2020-08-12_04-14-58.png)  
![图 14](images/P20_RNN_2020-08-12_10-48-59.png)  

## 门控循环单元GRU(Gated Recurrent Unit)

![图 9](images/P20_RNN_2020-08-12_03-58-58.png)  

重置门、更新门、候选隐含状态
![图 13](images/P20_RNN_2020-08-12_10-35-27.jpg)  
