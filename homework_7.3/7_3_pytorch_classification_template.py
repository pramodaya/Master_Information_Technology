import os
import pickle
import time
from collections import Counter
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (10, 7)  # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset_7feat.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/quick/2022-10-28-3901086A-802B-4C40-B45D-06CEC70F6B72.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, X_price, self.labels = pickle.load(fp)

        # Labels = [
        #     label_brands, ["BMW", "Audi", ...]
        #     label_fuel,
        #     label_transmission,
        #     label_seller_type
        # ]
        # X = list(zip(
        #     x_brands, [BMW, Audi, ...] => [0, 1 .. ]
        #     x_fuel,  [Diesel, Gasoline .. ]
        #     x_transmission, [Automatic, Manual]
        #     x_seller_type, [Dealer, Person]

        #     x_year, [2000, ..]
        #     x_km_driven, [200_000, .. ]
        #     x_owner [2, .. ]
        # ))

        self.X = np.concatenate((np.array(self.X), np.array(X_price)[:, np.newaxis]), axis=1)
        self.X_c = np.concatenate((self.X[:, :2], self.X[:, 3:4]), axis=1)

        self.Y = self.X[:, 2]  # y_transmission, [Automatic, Manual]
        self.Y_labels = self.labels[2]
        self.Y_len = len(self.Y_labels)
        self.labels.pop(1)

        self.X = np.array(self.X[:, 4:], dtype=np.float)
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),  # x_year, x_km_driven, x_owner, x_price
            torch.LongTensor(self.X_c[idx]),  # x_brands, x_fuel, x_seller_type
            torch.LongTensor(np.expand_dims(self.Y[idx], axis=-1))  # y_transmission, [Automatic, Manual]
        )


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=1)
        )

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(num_embeddings=2, embedding_dim=2),
            torch.nn.Embedding(num_embeddings=4, embedding_dim=2),
            torch.nn.Embedding(num_embeddings=4, embedding_dim=2)
        ])
        for i in range(1):
            self.embs.append(
                torch.nn.Embedding(embedding_dim=4,
                                   num_embeddings=len(dataset_full.labels[i]))
            )

    def forward(self, x, x_classes):
        x = torch.cat([x, self.embs[0](x_classes[:, 0]), self.embs[1](x_classes[:, 1]), self.embs[2](x_classes[:, 2])],
                      dim=-1)
        x = self.layers(x)
        return x


class LossBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        return -torch.mean(y * torch.log(y_prim) + (1 - y) * torch.log(1 - y_prim))


model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = LossBCE()

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
f1_plot_train = []
f1_plot_test = []
conf_matrix_train = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
conf_matrix_test = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        conf_matrix = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

        if dataloader == dataloader_train:
            torch.set_grad_enabled(True)
            model = model.train()
        else:
            torch.set_grad_enabled(False)
            model = model.eval()

        for x, x_classes, y in dataloader:

            y_prim = model.forward(x, x_classes)
            loss = loss_fn.forward(y_prim, y)

            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            acc = torch.mean(torch.eq(y, torch.argmax(y_prim, dim=-1)).type(torch.float))
            accs.append(acc)

        f1 = 0  # TODO

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            f1_plot_train.append(np.mean(f1))
            conf_matrix_train = conf_matrix
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            f1_plot_test.append(np.mean(f1))
            conf_matrix_test = conf_matrix

    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]} '
        f'f1_train: {f1_plot_train[-1]} '
        f'f1_test: {f1_plot_test[-1]} ')

    if epoch % 10 == 0 or 1:
        plt.tight_layout(pad=0)
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.tight_layout(pad=5)
        ax1 = axes[0, 0]
        ax1.set_title("Loss")
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[1, 0]
        ax1.set_title("Acc")
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax2 = ax1.twinx()
        ax2.plot(f1_plot_train, 'b-', label='f1_train')
        ax2 = ax1.twinx()
        ax2.plot(f1_plot_test, 'g-', label='f1_test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc.")

        ax1 = axes[0, 1]
        ax1.set_title("Train Conf.Mat")
        ax1.imshow(conf_matrix_train, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        ax1.set_xticklabels(dataset_full.Y_labels)
        ax1.set_yticklabels(dataset_full.Y_labels)
        ax1.set_yticks(np.arange(dataset_full.Y_len))
        ax1.set_xticks(np.arange(dataset_full.Y_len))
        for x in range(dataset_full.Y_len):
            for y in range(dataset_full.Y_len):
                ax1.annotate(
                    str(conf_matrix_train[x, y]),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor='black',
                    color='white'
                )
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        ax1 = axes[1, 1]
        ax1.set_title("Test Conf.Mat")
        ax1.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.get_cmap('Greys'))
        ax1.set_xticklabels(dataset_full.Y_labels)
        ax1.set_yticklabels(dataset_full.Y_labels)
        ax1.set_yticks(np.arange(dataset_full.Y_len))
        ax1.set_xticks(np.arange(dataset_full.Y_len))
        for x in range(dataset_full.Y_len):
            for y in range(dataset_full.Y_len):
                ax1.annotate(
                    str(conf_matrix_test[x, y]),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    backgroundcolor='black',
                    color='white'
                )
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        plt.show()
