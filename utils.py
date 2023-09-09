import numpy as np
    
class ReLU():
    def __call__(self, x):
        x[x < 0] = 0
        return x
    def d(self, x):
        return (x > 0).astype(np.float32)
    
class Id():
    def __call__(self, x):
        return x
    def d(self, x):
        return np.ones_like(x)



def loss(h, target):
    return (h - target)**2

def d_loss(h, target):
    return 2 * (h - target)