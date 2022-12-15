import matplotlib.pyplot as plt
import numpy as np

w_1 = 0
w_2 = 0
b = 0

X = np.array([
    [80, 5],
    [70, 10],
    [60, 2],
    [90, 15],
    [93, 20]
])

# real temp
Y = np.array([25, 20, 32, 21, 3])


def normalize(Z):
    Z_min = np.min(Z, axis=0)
    Z_max = np.min(Z, axis=0)
    Z_norm = (Z - 0.5 * (Z_max + Z_min)) / (Z - 0.5 * (Z_max - Z_min))
    return Z_norm, Z_min, Z_max


X_norm, X_min, X_max = normalize(X)
Y_norm, Y_min, Y_max = normalize(Y)
print(X_norm)
print(Y_norm)


def model(x_hum, x_win, w_1, w_2, b):
    y_hat = x_hum * w_1 + x_win * w_2 + b
    return y_hat


def loss(y_hat, y):
    loss_val = np.sum(np.abs(y_hat - y))
    return loss_val


def d_loss(y_hat, y):
    result = (y_hat - y + 1e-8) / np.abs(y_hat - y + 1e+8)
    return result


def d_w1_model(x_hum, x_win, w_1, w_2, b):
    return x_hum


def d_w1_loss(y_hat, y, x_hum, x_win, w_1, w_2, b):
    result = d_loss(y_hat, y) * d_w1_model(x_hum, x_win, w_1, w_2, b)
    return result


def d_w2_model(x_hum, x_win, w_1, w_2, b):
    return x_hum


def d_b_model(x_hum, x_win, w_1, w_2, b):
    return 1


def d_b_loss(y_hat, y, x_hum, x_win, w_1, w_2, b):
    result = d_loss(y_hat, y) * d_b_model(x_hum, x_win, w_1, w_2, b)
    return result


def d_w2_loss(y_hat, y, x_hum, x_win, w_1, w_2, b):
    result = d_loss(y_hat, y) * d_w2_model(x_hum, x_win, w_1, w_2, b)
    return result


alpha = 0.01
losses = []
for epoch in range(1, 10):
    loss_epoh = []
    for simple_idx in range(len(X_norm)):
        X_sample = X_norm[simple_idx]
        Y_sample = Y_norm[simple_idx]
        x_hum = X_sample[0]
        x_win = X_sample[1]
        y_hat = model(x_hum, x_win, w_1, w_2, b)
        loss_value = loss(y_hat, Y_sample)

        d_w1 = d_w1_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
        w_1 = w_1 - d_w1 * alpha

        d_w2 = d_w2_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
        w_2 = w_2 - d_w2 * alpha

        d_b = d_b_loss(y_hat, Y_sample, x_hum, x_win, w_1, w_2, b)
        b = b - d_b * alpha

        loss_epoh.append(loss_value)
    losses.append(np.mean(loss_epoh))
    print(np.mean(loss_epoh))

plt.plot(losses)
plt.show()
