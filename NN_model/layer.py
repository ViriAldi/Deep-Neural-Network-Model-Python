import numpy as np


class Layer:
    def __init__(self, n, n_prev, num, activation, derivative):
        self.num = num
        self.activation = activation
        self.derivative = derivative
        self.size = n
        self.prev = n_prev
        self.W = np.zeros((n, n_prev))
        self.B = np.zeros((n, 1))
        self.dW = np.zeros((n, n_prev))
        self.dB = np.zeros((n, 1))
        self.A = None
        self.Z = None
        self.dA = None
        self.dZ = None

    def forward_propagate(self, A_prev):
        self.Z = self.W @ A_prev + self.B
        self.A = self.activation(self.Z)

    def backward_propagate(self, dA, A_prev, m, alpha):
        self.dA = dA
        self.dZ = self.dA * self.derivative(self.Z)
        self.dB = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)
        self.dW = (1 / m) * self.dZ @ A_prev.T + (1 / m) * alpha * self.W

        return self.W.T @ self.dZ
