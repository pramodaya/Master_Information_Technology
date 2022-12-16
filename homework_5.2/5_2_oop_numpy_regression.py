import abc
import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7)  # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


class Dataset:
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
        #     x_brands,
        #     x_fuel,
        #     x_transmission,
        #     x_seller_type,
        #     x_year,
        #     x_km_driven,
        #     x_owner
        # ))

        self.X = np.array(self.X)
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std

        self.Y = np.array(self.Y)
        Y_mean = np.mean(self.Y)
        Y_std = np.std(self.Y)
        self.Y = (self.Y - Y_mean) / Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), self.Y[idx]


class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if len(self) < self.idx_batch:
            raise StopIteration()
        self.idx_start = self.idx_batch * self.batch_size + self.idx_start
        self.idx_end = self.idx_start + self.batch_size
        X, Y = self.dataset[self.idx_start: self.idx_end]
        Y = np.expand_dims(y, axis=-1)
        self.idx_batch += 1
        return X, Y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class Module(abc.ABC):
    def __init__(self):
        super().__init__()
        self.parameters = []

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self):
        pass

    def parameters(self):
        return self.parameters

    def __call__(self, x):
        return self.forward(x)


class LayerLinear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = Variable(
            value=np.random.random(size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b = Variable(
            value=np.random.random(size=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.parameters.append(self.W)
        self.parameters.append(self.b)
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.squeeze(
                self.W.value.T @ np.expand_dims(x.value, axis=2),
                axis=2
            ) + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += (
                np.expand_dims(self.x.value, axis=2) @
                np.expand_dims(self.output.grad, axis=1)
        )
        self.x.grad += np.squeeze(
            self.W.value @
            np.expand_dims(self.output.grad, axis=2),
            axis=2
        )


class LayerSigmoid(Module):
    def __init__(self):
        super().__init__()
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1.0 / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += (
                np.exp(-self.x.value) /
                (1 + np.exp(-self.x.value) ** 2)
        ) * self.output.grad


class LayerReLU(Module):
    def __init__(self):
        super().__init__()
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = x
        # TODO
        return self.output

    def backward(self):
        # TODO
        pass


class LossMAE(Module):
    def __init__(self):
        super().__init__()
        self.y = None
        self.y_prim = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = None
        self.y_prim = None
        loss = np.mean(np.abs(y.value-y_prim.value))
        return loss

    def backward(self):
        self.y =y
        self.y_prim = self.y_prim
        loss = np.mean(np.abs(self.y.value - self.y_prim.value)/np.abs(self.y.value - self.y_prim.value))
        return loss



class Model(Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            LayerLinear(in_features=7, out_features=5),
            LayerSigmoid(),
            LayerLinear(in_features=5, out_features=3),
            LayerSigmoid(),
            LayerLinear(in_features=3, out_features=1)
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            variables.extend(layer.parameters)
        return variables


class LayerEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.x: Variable = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_matrix = Variable(np.random.random((num_embeddings, embedding_dim)) - 0.5)
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x


    def backward(self):
        self.emb_matrix.grad += 0


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W = W -dW ** alpha
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate


    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMAE()

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, y in dataloader:

            y_prim = model.forward(Variable(x))
            loss = loss_fn.forward(Variable(y),y_prim )

            losses.append(loss)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

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
