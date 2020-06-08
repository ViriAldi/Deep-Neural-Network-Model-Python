import numpy as np


def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))


def tanh(matrix):
    return np.tanh(matrix)


def relu(matrix):
    return max(matrix, 0)


def leaky_relu(matrix):
    return max(matrix, -0.01 * matrix)
