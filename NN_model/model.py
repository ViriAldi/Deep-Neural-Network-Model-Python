from NN_model.layer import Layer
from NN_model.activations import *


class MultiLayerPerceptronClassifier:
    def __init__(self, layer_dims, activations):
        self.layers = []
        self.dims = layer_dims
        self.activations = activations
        self.L = layer_dims.shape[0]
        self.inputs = None
        self.loss = None

    def fit(self, X, y):
        self.inputs = X
        self.init_layers()
        self.init_weights()

    def init_layers(self):
        for lay in range(self.L):
            if lay == 0:
                n_prev = self.inputs.shape[0]
            else:
                n_prev = self.layers[lay - 1].shape[0]

            n = self.dims[lay]
            act = self.activations[lay]

            layer = Layer(n, n_prev, lay, act, DERIVATIVE[act])
            self.layers.append(layer)

        self.layers.append(Layer(1, self.dims[-1], self.L, sigmoid, sigmoid_derivative))

    def init_weights(self):
        for lay in self.layers:
            lay.W = np.random.randn(*lay.W.shape) * (1 / np.sqrt(lay.W.shape[0]))
            lay.B = np.zeros(*lay.B.shape)

    def gradient_descend(self, learning_rate, num_iter):
        self.propagate()

    def propagate(self):
        self.full_forward_propagation()

    def forward_propagation(self):
        for lay in range(self.L + 1):
            if lay == 0:
                A_prev = self.inputs
            else:
                A_prev = self.layers[lay - 1].A

            self.layers[lay].forward_propagate(A_prev)

        return self.layers[-1].A

    def compute_loss(self, y):
        y_hat = self.layers[-1].A
        self.loss = (1 / y.shape[0]) * np.sum((1 - y) * np.log(1 - y_hat) + y * np.log(y_hat))

    def full_forward_propagation(self, y):
        self.forward_propagation()
        self.compute_loss(y)

