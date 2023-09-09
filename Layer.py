import numpy as np

class Layer:
    def __init__(self, in_d, out_d, activation):
        self.z = np.zeros((out_d, 1))
        self.h = np.zeros((out_d, 1))

        self.W = np.random.uniform(-(np.sqrt(6)/np.sqrt(in_d + out_d)), (np.sqrt(6)/np.sqrt(in_d + out_d)), (out_d, in_d))
        self.b = np.zeros((out_d, 1))

        self.activation = activation

    def __call__(self, x):
        self.x = x

        self.z = np.einsum('ij,bjk->bik', self.W, self.x) + self.b
        self.h = self.activation(self.z)

        return self.h
    
    def backward(self, prev_g, prev_h):
        g = prev_g

        dh_dz = self.activation.d(self.z)
        g = g * dh_dz

        dz_dW = prev_h

        #W
        dl_dW = np.einsum('bij,bkj->bik', g, dz_dW)
        dl_dW = np.mean(dl_dW, axis=0)

        #b
        dz_db = np.ones_like(dh_dz)

        dl_db = g * dz_db
        dl_db = np.mean(dl_db, axis=0) #batch mean

        #next
        dz_dh = self.W
        next_g = np.einsum('ij,bik->bjk', dz_dh, g)


        return dl_dW, dl_db, next_g