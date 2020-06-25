from NN_model.layer import Layer
from NN_model.activations import *


class MultiLayerPerceptronClassifier:
    def __init__(self, layer_dims, activations, learning_rate=0.1, num_iter=1000, alpha=0):
        self.layers = []
        self.dims = layer_dims
        self.activations = activations
        self.L = len(layer_dims)
        self.inputs = None
        self.loss = None
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.alpha = alpha
        self.loss_curve = []

    def fit(self, X, y):
        self.inputs = X
        self.init_layers()
        self.init_weights()
        self.gradient_descend(self.learning_rate, self.num_iter, y)

    def init_layers(self):
        for lay in range(self.L):
            if lay == 0:
                n_prev = self.inputs.shape[0]
            else:
                n_prev = self.layers[lay - 1].size

            n = self.dims[lay]
            act = self.activations[lay]

            layer = Layer(n, n_prev, lay, act, DERIVATIVE[act])
            self.layers.append(layer)

        self.layers.append(Layer(1, self.dims[-1], self.L, sigmoid, sigmoid_derivative))

    def init_weights(self):
        for lay in self.layers:
            lay.W = np.random.randn(*lay.W.shape) * 2 / np.sqrt(lay.size)
            lay.B = np.zeros(lay.B.shape)

    def gradient_descend(self, learning_rate, num_epochs, y):
        for step in range(num_epochs):
            self.propagate(y)
            for lay in range(self.L + 1):
                layer = self.layers[lay]
                layer.W = layer.W - learning_rate * layer.dW
                layer.B = layer.B - learning_rate * layer.dB

            self.loss_curve.append(self.loss)
            if step % 50 == 0:
                print(f"Loss after {step}s iteration: ", self.loss)

    def propagate(self, y):
        self.full_forward_propagation(y)
        self.full_backward_propagation(y)

    def forward_propagation(self, inputs):
        for lay in range(self.L + 1):
            if lay == 0:
                A_prev = inputs
            else:
                A_prev = self.layers[lay - 1].A

            self.layers[lay].forward_propagate(A_prev)

        return self.layers[-1].A

    def compute_loss(self, y):
        y_hat = self.layers[-1].A
        self.loss = - (1 / y.shape[1]) * (np.sum((1 - y) * np.log(1 - y_hat) + y * np.log(y_hat))) + self.compute_l2(self.alpha, y.shape[1])

    def compute_l2(self, alpha, m):
        penalty = 0
        for layer in self.layers:
            penalty += np.sum(layer.W**2)

        return (1 / m) * (alpha / 2) * penalty

    def full_forward_propagation(self, y):
        self.forward_propagation(inputs=self.inputs)
        self.compute_loss(y)

    def compute_dAL(self, y):
        al = self.layers[-1].A
        return (1 / y.shape[1]) * (al - y) / (al - al**2)

    def backward_propagation(self, d_AL, m):
        da = d_AL
        for lay in reversed(range(self.L + 1)):
            if lay == 0:
                prev = self.inputs
            else:
                prev = self.layers[lay - 1].A

            da = self.layers[lay].backward_propagate(da, prev, m, self.alpha)

    def full_backward_propagation(self, y):
        m = y.shape[0]
        da = self.compute_dAL(y)

        self.backward_propagation(da, m)

    def predict(self, inputs):
        return self.forward_propagation(inputs)
