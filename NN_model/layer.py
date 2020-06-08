import numpy as np


class Layer:
    def __init__(self, n, n_prev, num, activation):
        self.num = num
        self.activation = activation
        self.size = n
        self.prev = n_prev
        self.W = np.zeros((n, n_prev))
        self.B = np.zeros((n, 1))
        self.W = np.zeros((n, n_prev))
        self.B = np.zeros((n, 1))
        self.A = None
        self.Z = None
        self.dA = None
        self.dZ = None

    def forward_propagate(self, A_prev):
