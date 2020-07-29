import math

import numpy as np
import pandas as pd

# 读取训练集，台湾/香港的繁体中文一般为big5编码
data = pd.read_csv("./resource/train.csv", encoding="big5")
# 去除前3列无用数据，iloc第一个参数表示行筛选，第二个参数表示列筛选
# 注意pandas默认会将第一行数据作为标题，所以不需要处理
data = data.iloc[:, 3:]
# 将下雨量为NR的数据置为0
data[data == 'NR'] = 0

# 将pandas DataFrame转为numpy数组
raw_data = np.array(data)

# 将原始数据转为易处理的格式，原来是4320行(12月*20天*18个特征)，24(小时)列
# 转成每月为18(特征)行，480列(20天*24小时)的数据
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        # 18个特征的重复列各都合成为一列
        sample[:, day * 24:(day + 1) * 24] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :]
        month_data[month] = sample

# 再根据题目需要，将前9个小时的18个feature展开，看作为一共有18*9个feature，第10个小时的pm2.5作为标签，则一共有12*471份数据
# 该数据被重复使用，比如，0-8，1-9，2-10小时作为feature，而9，10，11小时作为标签
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
# 循环了12*(20*24-9)=12*471次
for month in range(12):
    for day in range(20):
        for hour in range(24):
            # 最大值为day * 24 + hour + 9=479(最大列下标为479)，其中day取值为19，代入即可得到hour等于14
            # 所以当day等于19的时候，hour不得大于14，否则会发生数组越界
            if day == 19 and hour > 14:
                continue
            # 将9个小时的18个特征展开成一行数据，这里-1表示自动计算展开后的列数
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     -1)
            # 行下标为9的数据为pm2.5
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
# 归一化，axis=0表示求该列所有数的均值和标准差（可看作合并所有行为一行）
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    x[i] = (x[i] - mean_x) / std_x

# 将已有数据分为训练数据和校验数据
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# 维度，加1原因是bias
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 0.000001
iter_time = 1000

w_temp = np.copy(w)


def calculate_gradient():
    # 除以12*471原因是有12*471份数据相加了，这里求均值即可
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 12 / 471)
    if t % 100 == 0:
        print(f"{t=},{loss=}")
    # 通过复合函数求偏导得到此公式，可以用数量比较少的w和x(数据)来校验这个公式
    return 2 * np.dot(x.transpose(), np.dot(x, w) - y)


# 普通梯度下降法(最原始，也叫批量梯度下降BGD)
print("普通梯度效果")
for t in range(iter_time):
    gradient = calculate_gradient()
    # 更新w
    w -= learning_rate * gradient

# adagrad
print("adagrad效果")
w = w_temp
adagrad = np.zeros([dim, 1])
# eps是为了防止adagrad为零
eps = 0.0000000001
learning_rate = 2

for t in range(iter_time):
    gradient = calculate_gradient()
    # adagrad是前面所有(包括当前)积分的均方根(将N个项的平方和除以N后开平方)
    # 两个乘号意思是乘方，即2个gradient相乘
    adagrad += gradient ** 2
    w -= learning_rate * gradient / np.sqrt(adagrad + eps)
