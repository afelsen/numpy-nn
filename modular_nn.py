import numpy as np
from activation import Sigmoid, ReLU, Id
from loss import MSE
from optim import SGD, Adam
import matplotlib.pyplot as plt

from Data import Data
from Layer import Layer
    
class NN:
    def __init__(self, in_dim=1, out_dim=1):
        self.lr = 0.001
        
        d1 = 32; d2 = 64; d3 = 32

        self.layers = [
            Layer(in_dim, d1, activation=ReLU()),
            Layer(d1, d2, activation=ReLU()),
            Layer(d2, d3, activation=ReLU()),
            Layer(d3, out_dim, activation=Id())
        ]

        self.w_updates = [None] * len(self.layers)
        self.b_updates = [None] * len(self.layers)

    def forward(self, x):
        self.x = x

        for layer in self.layers:
            x = layer(x)

        return x
    
    def backward(self, target, loss_fn):
        hs = [self.x] + [layer.h for layer in self.layers]
        hs = hs[::-1]

        dl_dh = loss_fn.d(hs[0], target)
        prev_g = dl_dh

        for i, layer in enumerate(self.layers[::-1]):
            dl_dW, dl_db, prev_g = layer.backward(prev_g, hs[i+1])

            self.w_updates[i] = dl_dW
            self.b_updates[i] = dl_db

def train(nn, data, epoch=10000):
    optimizer = Adam(nn.layers, lr=nn.lr)
    loss = MSE()
    
    for i in range(epoch):
        x, label = data.__iter__()
        h = nn.forward(x)
        l = loss(h, label)
        nn.backward(label, loss)

        optimizer.step(nn.w_updates, nn.b_updates)

        print(f"{i}: {l}")

        if i % 100 == 0:
            eval(nn, data, 10)
            plt.pause(0.01)
            plt.clf()

    plt.show()

def eval(nn, data, n=10):
    xs = []
    ys = []
    for i in range(n):
        x, label = data.__iter__()
        h = nn.forward(x)

        xs += list(x[:,0,0])
        ys += list(h[:,0,0])
    
    xs = np.array(xs)
    ys = np.array(ys)

    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    plt.scatter(xs, ys, label="Prediction")
    plt.plot(xs, data.f(xs), label="Ground Truth", color="red")

    plt.title("Evaluation")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend(loc="upper left")

if __name__ == "__main__":
    nn = NN()
    data = Data()

    train(nn, data, 50000)
    eval(nn, data)