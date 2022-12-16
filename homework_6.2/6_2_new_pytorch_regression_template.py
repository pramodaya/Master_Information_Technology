import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (12, 7) # size of window
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
            self.X, self.Y, self.labels = pickle.load(fp)

        # Labels = [
        #     label_brands,
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
        self.X = np.array(self.X)
        self.X_c = self.X[:, :4]

        self.X = np.array(self.X[:, 4:], dtype=np.float)
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std

        self.Y = np.array(self.Y, dtype=np.float)
        Y_mean = np.mean(self.Y)
        Y_std = np.std(self.Y)
        self.Y = (self.Y - Y_mean) / Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return(
                torch.FloatTensor(self.X[idx]),
                torch.LongTensor(self.X_c[idx]),
                torch.FloatTensor(np.expand_dims(self.Y[idx], axis=-1)))


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

# TODO Model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=3+12, out_features=5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=5, out_features=3),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=3, out_features=1)
        )

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(embedding_dim=3, num_embeddings=len(dataset_full.labels[0])),
            torch.nn.Embedding(embedding_dim=3, num_embeddings=len(dataset_full.labels[1])),
            torch.nn.Embedding(embedding_dim=3, num_embeddings=len(dataset_full.labels[2])),
            torch.nn.Embedding(embedding_dim=3, num_embeddings=len(dataset_full.labels[3]))
        ])

    def forward(self, x, x_c):
        x_e = []
        for i, emb in enumerate(self.embs):
            x_e.append(
                emb.forward(x_c[:, i])
            )
        x_e = torch.cat(x_e, dim=-1)
        x = torch.cat([x, x_e], dim=-1)
        y_prim = self.layers.forward(x)
        return y_prim

class LossMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return torch.mean((y - y_prim) ** 2)

model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = LossMSE() # TODO Loss

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, x_c, y in dataloader:

            y_prim = model.forward(x, x_c)
            loss = loss_fn.forward(y_prim, y)

            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
