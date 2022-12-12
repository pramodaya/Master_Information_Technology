import numpy as np

X_raw = np.array([
    [2003, 680_000],
    [2007, 519_000],
    [2010, 471_300],
    [2009, 633_000],
    [2014, 311_000]
], dtype=np.float)

Y_raw = np.array([
    [12_500],
    [13_500],
    [19_000],
    [18_000],
    [20_600]
], dtype=np.float)

print(X_raw[2])
print(X_raw[:, 1])
print(X_raw[0:5, 1])
print(X_raw[3, 0])

X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
Y_mean = np.mean(Y_raw, axis=0)
Y_std = np.std(Y_raw, axis=0)
Y = (Y_raw - Y_mean) / Y_std

w_1 = np.random.rand(2, 3)
b_1 = np.zeros(3)
w_2 = np.random.rand(3, 2)
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




# New Data with 3 variables


def model_threee_param(X, w_1, b_1, w_2, b_2, w_3, b_3):
    out_1 = linear(w_1, b_1, X)
    out_2 = sigmoid(out_1)
    out_3 = linear(w_2, b_2, out_2)
    out_4 = sigmoid(w_3, b_3, out_3)
    out_5 = linear(w_3, b_3, out_4)
    return out_5

def dw1_loss_three_param(X, w_1, b_1, w_2, b_2, w_3, b_3):
    y_prim = model_threee_param(X, w_1, b_1, w_2, b_2, w_3, b_3)


X_raw = np.array([
    [2003, 680_000, 4],
    [2007, 519_000, 4],
    [2010, 471_300, 4],
    [2009, 633_000, 2],
    [2014, 311_000, 4]
], dtype=np.float)

Y_raw = np.array([
    [12_500],
    [13_500],
    [19_000],
    [18_000],
    [20_600]
], dtype=np.float)

w_3 = np.random.rand(3, 1)
b_3 = np.zeros(1)

Y_prim = model_threee_param(X, w_1, b_1, w_2, b_2, w_3, b_3)
loss = loss_mae(Y_prim, Y)

print(f'Y_prim : {Y_prim}')
print(f'Y : {Y}')
