import numpy as np

class ReLU():
    def __call__(self, x):
        x[x < 0] = 0
        return x
    def d(self, x):
        return (x > 0).astype(np.float32)
    
class Sigmoid():
    def __call__(self, x):
        return self._sig(x)
    def _sig(self, x):
        return 1 / (1 + np.exp(-x))
    def d(self, x):
        return self._sig(x) * (1 - self._sig(x))

class Id():
    def __call__(self, x):
        return x
    def d(self, x):
        return np.ones_like(x)