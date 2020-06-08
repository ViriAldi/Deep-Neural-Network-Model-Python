import numpy as np


def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))


def sigmoid_derivative(matrix):
    return np.exp(-matrix) / (1 + np.exp(-matrix))**2


def tanh(matrix):
    return np.tanh(matrix)


def tanh_derivative(matrix):
    return 4 * np.exp(2 * matrix) / (1 + np.exp(2 * matrix))**2


def relu(matrix):
    return np.maximum(matrix, 0)


def relu_derivative(matrix):
    return (matrix > 0).astype(int)


DERIVATIVE = {
    sigmoid: sigmoid_derivative,
    tanh: tanh_derivative,
    relu: relu_derivative
}
