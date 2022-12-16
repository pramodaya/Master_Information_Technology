import pickle

import matplotlib.pyplot as plt
import numpy as np

import requests

URL = "http://share.yellowrobot.xyz/quick/2022-10-14-66587525-E080-4622-96D5-FE08B43EF982.pkl"
response = requests.get(URL)
open("data.pkl", "wb").write(response.content)

with open('./data.pkl', 'rb') as fp:
    X, Y = pickle.load(fp)
    X = np.array(X)

w_1 = 0
w_2 = 0
w_3 = 0  # NEW WEIGHT
b = 0

x_hum = X[:, 0]  # Year
x_win = X[:, 1]  # Drived KMs
x_owner_count = X[:, 2]  # Owner count

print("X_HUM")
print(x_hum)
print("X_WIN")
print(x_win)
print("X_OWNER_COUNT")
print(x_owner_count)

def normalize(Z):
    Z_min = np.min(Z, axis=0)
    Z_max = np.min(Z, axis=0)
    Z_norm = (Z - 0.5 * (Z_max + Z_min)) / (Z - 0.5 * (Z_max - Z_min))
    return Z_norm, Z_min, Z_max


X, X_min, X_max = normalize(X)
Y, Y_min, Y_max = normalize(Y)
print(X)
print(Y)


def model(x_hum, x_win, x_owner_count, w_1, w_2, b):
    y_hat = x_hum * w_1 * 3 ** 2 + x_win * w_2 * 2 + x_owner_count * w_3 + b
    return y_hat


def d_w1_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    return x_hum * 3 * w_1


def d_w2_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    return x_win * 2 * w_2


def d_w3_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    return x_owner_count * w_3 * w_2


def d_b_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    return 1


def loss(y_hat, y):
    loss_val = np.sum(np.abs(y_hat - y))
    return loss_val


def d_loss(y_hat, y):
    result = (y_hat - y + 1e-8) / np.abs(y_hat - y + 1e+8)
    return result


def d_w1_loss(y_hat, y, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    result = d_loss(y_hat, y) * d_w1_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    return result


def d_w2_loss(y_hat, y, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    result = d_loss(y_hat, y) * d_w2_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    return result


def d_w3_loss(y_hat, y, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    result = d_loss(y_hat, y) * d_w3_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    return result


def d_b_loss(y_hat, y, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b):
    result = d_loss(y_hat, y) * d_b_model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    return result


alpha = 0.01
losses = []

all_d_w1 = []
all_d_w2 = []
all_d_w3 = []
all_d_b = []

for epoch in range(1, 100):

    y_hat = model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    loss_value = loss(Y, y_hat)

    d_w1 = d_w1_loss(Y, y_hat, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    all_d_w1.append(d_w1)
    d_w2 = d_w2_loss(Y, y_hat, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    all_d_w2.append(d_w2)
    d_w3 = d_w3_loss(Y, y_hat, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    all_d_w3.append(d_w3)
    d_b = d_b_loss(Y, y_hat, x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    all_d_b.append(d_b)

    losses.append(loss_value)

    w_1 = w_1 - np.mean(all_d_w1) * alpha
    w_2 = w_2 - np.mean(all_d_w2) * alpha
    w_3 = w_3 - np.mean(all_d_w3) * alpha
    b = b - np.mean(all_d_b) * alpha

    # for simple_idx in range(len(X_norm)):
    #     X_sample = X_norm[simple_idx]
    #     Y_sample = Y_norm[simple_idx]
    #     x_hum = X_sample[0]
    #     x_win = X_sample[1]
    #     y_hat = model(x_hum, x_win, x_owner_count, w_1, w_2, w_3, b)
    #     loss_value = loss(y_hat, Y_sample)
    #
    #     # d_w1 = d_w1_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
    #     # all_d_w1.append(d_w1)
    #     #
    #     # d_w2 = d_w2_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
    #     # all_d_w2.append(d_w2)
    #     #
    #     # d_b = d_b_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
    #     # all_d_b.append(d_b)
    #
    #     loss_epoh.append(loss_value)

    # w_1 = w_1 - np.mean(all_d_w1) * alpha
    # w_2 = w_2 - np.mean(all_d_w2) * alpha
    # b = b - np.mean(all_d_b) * alpha
    #
    # losses.append(np.mean(loss_epoh))
    # print(np.mean(loss_epoh))

plt.plot(losses)
plt.show()
