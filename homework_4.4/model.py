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
    Y = np.expand_dims(Y, axis=1)


def standardize(Z):
    Z_mu = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0)
    Z_n = (Z - Z_mu) / Z_std
    return Z_n, Z_mu, Z_std


X, X_mu, X_std = standardize(X)
Y, Y_mu, Y_std = standardize(Y)


# X values: km_driven, year_make, owners
# Y value = price

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


alpha = 0.01
losses = []
for epoch in range(1, 100):
    y_hat = 0  # TODO model prediction
    loss_val = 0  # TODO loss
    losses.append(loss_val)

    # TODO weights using SGD

plt.plot(losses)
plt.show()
