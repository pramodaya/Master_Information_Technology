import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.hub import download_url_to_file

path_dataset = './cardekho_regression.pkl'
if not os.path.exists(path_dataset):
    os.makedirs('../data', exist_ok=True)
    download_url_to_file(
        'http://share.yellowrobot.xyz/quick/2022-10-14-66587525-E080-4622-96D5-FE08B43EF982.pkl',
        path_dataset,
        progress=True
    )
with open(f'{path_dataset}', 'rb') as fp:
    X, Y = pickle.load(fp)
    X = np.array(X)


def standardize(Z):
    Z_mu = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0)
    Z_n = (Z - Z_mu) / Z_std
    return Z_n, Z_mu, Z_std


X, X_mu, X_std = standardize(X)
Y, Y_mu, Y_std = standardize(Y)

W = np.array([
    [0],
    [0],
    [0]
])

b = np.array([0])

Y = np.expand_dims(Y, axis=1)

val = np.array([1, 2, 3])
val_expand_1 = np.expand_dims(val, axis=0)
val_expand_2 = np.expand_dims(val, axis=1)

print(val, val_expand_1, val_expand_2)


# linear function
def linear(X, W, b):
    # np.dot and np.matual
    y_hat = np.squeeze(np.expand_dims(X, axis=1) @ W, axis=1) + b
    return y_hat


def dw_linear(X, W, b):
    return X


def model_linear(X, W, b):
    y_hat = 0
    return y_hat


def db_linear(X, W, b):
    return 1


def dx_liner(X, W, b):
    return W


def loss(y_hat, y):
    loss_val = np.sum(np.abs(y_hat - y))
    return loss_val


def dw_loss(X, W, b, y_hat, y):
    return d_y_hat_loss(y_hat, y) @ dw_linear(X, W, b)


def db_loss(X, W, b, y_hat, y):
    return d_y_hat_loss(y_hat, y) @ db_linear(X, W, b)


def d_y_hat_loss(y_hat, y):  # loss/d_y_hat
    result = (y_hat - y + 1e-8) / np.abs(y_hat - y + 1e-8)
    return result


alpha = 0.01
losses = []
for epoch in range(1, 100):
    y_hat = linear(X, W, b)
    loss = loss(y_hat, Y)
    dw = dw_loss(X, W, b, y_hat, Y)
    W = W - dw * alpha

plt.plot(losses)
plt.show()
