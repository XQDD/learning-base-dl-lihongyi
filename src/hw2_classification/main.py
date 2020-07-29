import numpy as np
import pandas as pd

x_data = pd.read_csv("./resource/X_train", encoding="big5").iloc[:, 1:]
x = np.array(x_data)

y_data = pd.read_csv("./resource/Y_train", encoding="big5").iloc[:, 1:]
y = np.array(y_data)

X_train = x
Y_train = y
data_dim = X_train.shape[1]


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def accuracy(y_pred, y_label):
    return ((y_label - y_pred.reshape(y_label.shape)) == 0).sum() / y_label.size


"""discriminative"""
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    x[i] = (x[i] - mean_x) / (std_x + 1e-8)

x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
w = np.zeros([x.shape[1], 1])

learning_rate = 0.001
iter_time = 101

for t in range(iter_time):
    w -= learning_rate / np.sqrt(t + 1) * np.dot(x.T, sigmoid(np.dot(x, w)) - y)
    if t % 100 == 0:
        print(accuracy(np.round(sigmoid(np.dot(x, w))), y))

"""generative【*未掌握】"""
x = x[:, 1:]
# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for _x in X_train_0:
    cov_0 += np.dot(np.transpose([_x - mean_0]), [_x - mean_0]) / X_train_0.shape[0]
for _x in X_train_1:
    cov_1 += np.dot(np.transpose([_x - mean_1]), [_x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) \
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])


def _f(X, w, b):
    return sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)


# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(accuracy(Y_train_pred, Y_train)))
