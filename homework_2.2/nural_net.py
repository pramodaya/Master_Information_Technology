import numpy as np
import pandas as pd


def act_fun(x):
    # the activation function 
    bias = -.5
    return 1 / (1 + np.exp(-x + bias))


def der_act_fun(x):
    # derivative of the activation function
    bias = -.5
    return np.exp(-x + bias) / np.square(1 + np.exp(-x + bias))


def one_hot(labels, n_classes):
    num_labels = len(labels)

    one_hot_labels = np.zeros([num_labels, n_classes])

    for i in range(num_labels):
        j = labels[i]
        one_hot_labels[i, j] = 1

    return one_hot_labels


class layer(object):
    def __init__(self, n_in, n_nodes, act_fun='sigmoid'):
        # scale to some reasonable random uniform distribution
        if act_fun == 'sigmoid':
            scale_fac = 4 * np.sqrt(6) / (np.sqrt(n_in + n_nodes))
            self.weights = scale_fac * (2 * np.random.rand(n_nodes, n_in) - 1)
        self.size = n_nodes
        self.wsum = np.zeros(n_nodes)
        self.deltas = np.zeros(n_nodes)
        self.activ = np.zeros(n_nodes)

    def activate(self, inp):
        self.wsum = self.weights.dot(inp)
        self.activ = act_fun(self.wsum)
        return self.activ


class neural_net(object):

    def __init__(self, n_in, nodes_per_layer):
        self.n_layers = len(nodes_per_layer)
        self.ni = n_in
        self.no = nodes_per_layer[self.n_layers - 1]
        self.layer_sizes = nodes_per_layer
        self.layers = []

        # create first layer (layer 0)
        self.layers += [layer(n_in, nodes_per_layer[0])]

        # create hidden layers and output layer
        for i in range(1, self.n_layers):
            self.layers += [layer(nodes_per_layer[i - 1], nodes_per_layer[i])]

    def activate(self, vec):

        for i in range(self.n_layers):
            vec = self.layers[i].activate(vec)
        return vec

    def backpropagation(self, activation, target, learning_rate):

        n_layers = self.n_layers
        layers = self.layers

        # calculate deltas for last layer
        layers[n_layers - 1].deltas = der_act_fun(layers[n_layers - 1].wsum) * (target - activation)

        # calculate deltas for hidden layers
        for l in range(n_layers - 2, 0, -1):
            layers[l].deltas = der_act_fun(layers[l].wsum) * (
                np.transpose(layers[l + 1].weights).dot(layers[l + 1].deltas))

        # update weights for last and hidden layers with the deltas
        for l in range(n_layers - 1, 1, -1):
            for i in range(layers[l].size):
                for j in range(layers[i - 1].size):
                    layers[l].weights[i, j] = layers[l].weights[i, j] + learning_rate * layers[l].deltas[i] * \
                                              layers[i - 1].activ[j]

        # update weights for first (zeroeth) layer
        for i in range(layers[0].size):
            for j in range(len(target)):
                layers[0].weights[i, j] = layers[0].weights[i, j] + learning_rate * layers[0].deltas[i] * target[j]

    def training_epoch(self, data, targets, learning_rate):

        n_examples = np.size(data, 1)

        avg_cost = 0

        # online learning
        for i in range(n_examples):
            activation = self.activate(data[i, :])
            avg_cost += sum(np.square(activation - targets[i, :]))
            self.backpropagation(activation, targets[i, :], learning_rate)

        avg_cost = avg_cost / n_examples
        return avg_cost

    def regularization(self):
        # regularization term for the neural network
        reg = 0
        for layer in self.layers:
            reg += sum(layer.weights ** 2)
        return reg

def train(nn, n_epochs=50000,
          learning_rate=.008,
          datafile='train.csv',
          nodes_per_layer=[5, 5],
          regularization=True,
          reg_param=.01):
    # xor test input
    raw_data = np.array([[0, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1]])

    # create array of target vectors  
    targets = one_hot(labels=raw_data[:, 0], n_classes=2)

    # data (examples stored in columns)
    data = raw_data[:, 1:]

    for n in range(n_epochs):
        avg_cost = nn.training_epoch(data, targets, learning_rate)
        print(n, avg_cost)


if __name__ == '__main__':
    # build neural net
    nn = neural_net(n_in=2, nodes_per_layer=[5, 5, 2])

    train(nn)
