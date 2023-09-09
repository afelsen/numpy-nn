import numpy as np

class ReLU():
    def __call__(self, x):
        x[x < 0] = 0
        return x
    def d(self, x):
        return (x > 0).astype(np.float32)
    
class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def d(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Id():
    def __call__(self, x):
        return x
    def d(self, x):
        return np.ones_like(x)