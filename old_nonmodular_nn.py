import numpy as np
import matplotlib.pyplot as plt

def sig(x):
    x[x < 0] = 0
    return x

def d_sig(x):
    return (x > 0).astype(np.float32)

def id(x):
    return x

def d_id(x):
    return np.ones_like(x)

def loss(h, target):
    return (h - target)**2

def d_loss(h, target):
    return 2 * (h - target)

class Data:
    def __init__(self):
        self.batch_size = 64
    def f(self, x):
        return np.sin(x)**2#x**3 + x**2 - x
    def __iter__(self):
        x = 10*np.array([2*(np.random.rand()-0.5) for i in range(self.batch_size)])[:, None, None]
        label = self.f(x)

        return x, label

class NN:
    def __init__(self, in_dim=1, out_dim=1):
        self.lr = 0.001

        d1 = d2 = 30

        self.h1 = np.zeros((d1, 1))
        self.h2 = np.zeros((d2, 1))
        self.h3 = np.zeros((out_dim, 1))

        self.z1 = np.zeros((d1, 1))
        self.z2 = np.zeros((d2, 1))
        self.z3 = np.zeros((out_dim, 1))

        self.W1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(1 + d1)), (np.sqrt(6)/np.sqrt(1 + d1)), (d1, in_dim))
        self.W2 = np.random.uniform(-(np.sqrt(6)/np.sqrt(d1 + d2)), (np.sqrt(6)/np.sqrt(d1 + d2)), (d2, d1))
        self.W3 = np.random.uniform(-(np.sqrt(6)/np.sqrt(1 + d2)), (np.sqrt(6)/np.sqrt(1 + d2)), (out_dim, d2))

        self.b1 = np.zeros((d1, 1))
        self.b2 = np.zeros((d2, 1))
        self.b3 = np.zeros((out_dim, 1))

    def forward(self, x):
        self.x = x

        self.z1 = np.einsum('ij,bjk->bik', self.W1, self.x) + self.b1#self.W1 @ self.x + self.b1
        self.h1 = sig(self.z1)
        self.z2 = np.einsum('ij,bjk->bik', self.W2, self.h1) + self.b2 #self.W2 @ self.h1 + self.b2
        self.h2 = sig(self.z2)
        self.z3 = np.einsum('ij,bjk->bik', self.W3, self.h2) + self.b3 #self.W3 @ self.h2 + self.b3
        self.h3 = id(self.z3)

        return self.h3        
    
    def backward(self, target):
        #weights
        dl_dh3 = d_loss(self.h3, target)

        #W3
        dh3_dz3 = d_id(self.z3)
        g3 = dh3_dz3 * 1

        dz3_dW3 = self.h2
 
        dl_dW3 = np.einsum('bij,bkj->bik', dl_dh3 * g3, dz3_dW3) #dl_dh3 * g3 @ dz3_dW3.T
        dl_dW3 = np.mean(dl_dW3, axis=0) #batch mean
        self.W3 = self.W3 - self.lr * dl_dW3

        #W2
        dz3_dh2 = self.W3
        dh2_dz2 = d_sig(self.z2)
        g2 = np.einsum('ij,bik->bjk', dz3_dh2, g3) * dh2_dz2 #dz3_dh2.T @ g3 * dh2_dz2
        
        dz2_dW2 = self.h1

        dl_dW2 = np.einsum('bij,bkj->bik', dl_dh3 * g2, dz2_dW2) #dl_dh3 * g2 @ dz2_dW2.T
        dl_dW2 = np.mean(dl_dW2, axis=0) #batch mean
        self.W2 = self.W2 - self.lr * dl_dW2
        
        #W1
        dz2_dh1 = self.W2
        dh1_dz1 = d_sig(self.z1)
        g1 = np.einsum('ij,bik->bjk', dz2_dh1, g2) * dh1_dz1 # dz2_dh1.T @ g2 * dh1_dz1

        dz1_dW1 = self.x

        dl_dW1 = np.einsum('bij,bkj->bik', dl_dh3 * g1, dz1_dW1) #dl_dh3 * g1 @ dz1_dW1.T
        dl_dW1 = np.mean(dl_dW1, axis=0) #batch mean
        self.W1 = self.W1 - self.lr * dl_dW1

        #biases
        #b3
        dz3_db3 = np.ones_like(dh3_dz3)

        dl_db3 = dl_dh3 * g3 * dz3_db3
        dl_db3 = np.mean(dl_db3, axis=0) #batch mean
        self.b3 = self.b3 - self.lr * dl_db3

        #b2
        dz3_db2 = np.ones_like(dh2_dz2)

        dl_db2 = dl_dh3 * g2 * dz3_db2
        dl_db2 = np.mean(dl_db2, axis=0) #batch mean
        self.b2 = self.b2 - self.lr * dl_db2

        #b1
        dz2_db1 = np.ones_like(dh1_dz1)

        dl_db1 = dl_dh3 * g1 * dz2_db1
        dl_db1 = np.mean(dl_db1, axis=0) #batch mean
        self.b1 = self.b1 - self.lr * dl_db1


def train(nn, data, epoch=10000):
    
    for i in range(epoch):
        x, label = data.__iter__()
        h = nn.forward(x)
        l = np.mean(loss(h, label))
        nn.backward(label)
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

    plt.scatter(xs, ys)

    plt.plot(xs, data.f(xs))
    

if __name__ == "__main__":
    nn = NN()
    data = Data()

    train(nn, data, 500000)
    eval(nn, data)