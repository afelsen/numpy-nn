import numpy as np

class SGD:
    def __init__(self, layers, lr=0.001):
        self.alpha = lr

        self.layers = layers[::-1]

    def step(self, w_updates, b_updates):
        for i, layer in enumerate(self.layers):
            layer.W -= self.alpha * w_updates[i]
            layer.b -= self.alpha * b_updates[i]

class Adam:
    def __init__(self, layers, lr=0.001):
        self.alpha = lr
        self.beta1 = 0.99
        self.beta2 = 0.999

        self.epsilon = 1e-8

        self.layers = layers[::-1]

        self.m_w = [np.zeros_like(layer.W) for layer in self.layers]
        self.m_b = [np.zeros_like(layer.b) for layer in self.layers]

        self.v_w = [np.zeros_like(layer.W) for layer in self.layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.layers]

    def step(self, w_updates, b_updates):
        for i, layer in enumerate(self.layers):
            #First moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * w_updates[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * b_updates[i]

            #Second moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * w_updates[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * b_updates[i]**2

            #Bias corrected first moment estimate
            m_w_hat = self.m_w[i] / (1 - self.beta1)
            m_b_hat = self.m_b[i] / (1 - self.beta1)

            #Bias corrected second moment estimate
            v_w_hat = self.v_w[i] / (1 - self.beta2)
            v_b_hat = self.v_b[i] / (1 - self.beta2)

            #Update parameters
            layer.W -= self.alpha * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.b -= self.alpha * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
