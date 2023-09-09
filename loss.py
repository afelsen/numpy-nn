import numpy as np

class MSE():
    def __call__(self, h, target):
        return np.mean((h - target)**2)
    def d(self, h, target):
        return 2 * (h - target)