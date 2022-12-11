import numpy as np

X_raw = np.array([
    [2010, 380_000],
    [2009, 419_000],
    [2011, 251_700],
    [2010, 633_500],
    [2002, 611_000]
], dtype=np.float)

Y_raw = np.array([
    [22_500],
    [17_500],
    [28_000],
    [16_000],
    [5_600]
], dtype=np.float)

X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
Y_mean = np.mean(Y_raw, axis=0)
Y_std = np.std(Y_raw, axis=0)
Y = (Y_raw - Y_mean) / Y_std

w_1 = np.random.rand(2, 3)
b_1 = np.zeros(3)
w_2 = np.random.rand(3, 1)
b_2 = np.zeros(1)


def linear(w, b, x):
    out_1 = w.transpose() @ x[:, :, np.newaxis]
    return out_1[:, :] + b


def dx_linear(w, b, x):
    return w


def dw_linear(w, b, x):
    return x


def db_linear(w, b, x):
    return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dx_sigmoid(x):
    return sigmoid(x)(1 - sigmoid(x))


def model(X, w_1, b_1, w_2, b_2):
    out_1 = linear(w_1, b_1, X)
    out_2 = sigmoid(out_1)
    out_3 = linear(w_2, b_2, out_2)
    return out_3


def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim - y))


def dy_prim_loss_mae(y_prim, y):
    return (y_prim + 1e-8) / (np.abs(y_prim) + 1e-8)


def dw1_loss(X, w_1, b_1, w_2, b_2):
    y_prim = model(X, w_1, b_1, w_2, b_2)


Y_prim = model(X, w_1, b_1, w_2, b_2)
loss = loss_mae(Y_prim, Y)

print(f'Y_prim : {Y_prim}')
print(f'Y : {Y}')
