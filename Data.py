import numpy as np

class Data:
    def __init__(self):
        self.batch_size = 64
    def f(self, x):
        return np.sin(x)**2#x**3 + x**2 - x
    def __iter__(self):
        x = 10*np.array([2*(np.random.rand()-0.5) for i in range(self.batch_size)])[:, None, None]
        label = self.f(x)

        return x, label