from NN_model.layer import Layer
from NN_model.activations import *
from math import ceil


class MultiLayerPerceptronClassifier:
    def __init__(self, layer_dims, activations, learning_rate=0.1, num_epochs=1000, alpha=0, batch_size=64,
                 optimizer="adam", beta1=0.9, beta2=0.999, epsilon=10**-8):
        self.layers = []
        self.dims = layer_dims
        self.activations = activations
        self.L = len(layer_dims)

        self.inputs = None
        self.y = None
        self.loss = None

        self.num_epochs = num_epochs
        self.alpha = alpha
        self.loss_curve = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.mini_batches = []

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def fit(self, X, y):
        self.inputs = X
        self.y = y
        self.init_layers()
        self.init_weights()
        self.gradient_descend(self.learning_rate, self.num_epochs)

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

    def _init_adam(self):
        for layer in self.layers:
            layer.Vel_W = np.zeros(layer.W.shape)
            layer.Sqr_W = np.zeros(layer.W.shape)
            layer.Vel_b = np.zeros(layer.B.shape)
            layer.Sqr_b = np.zeros(layer.B.shape)

    def _shuffle_inputs(self):
        shuffled = list(np.random.permutation(self.inputs.shape[1]))
        self.inputs = self.inputs[:, shuffled]
        self.y = self.y[:, shuffled]

        for k in range(ceil(self.inputs.shape[1] / self.batch_size)):
            mini_batch_x = self.inputs[:, k * self.batch_size:(k+1) * self.batch_size]
            mini_batch_y = self.y[:, k * self.batch_size:(k + 1) * self.batch_size]
            self.mini_batches.append((mini_batch_x, mini_batch_y))

    def gradient_descend(self, learning_rate, num_epochs):
        if self.optimizer == "adam":
            self._init_adam()
            adam_t = 1
        for epoch in range(num_epochs):
            self._shuffle_inputs()
            for mini_batch in self.mini_batches:
                self.propagate(*mini_batch)
                for layer in self.layers:
                    if self.optimizer == "adam":
                        layer.Vel_W = self.beta1 * layer.Vel_W + (1 - self.beta1) * layer.dW
                        layer.Vel_b = self.beta1 * layer.Vel_b + (1 - self.beta1) * layer.dB
                        layer.Sqr_W = self.beta2 * layer.Sqr_W + (1 - self.beta2) * layer.dW**2
                        layer.Sqr_b = self.beta2 * layer.Sqr_b + (1 - self.beta2) * layer.dB**2

                        layer.W -= learning_rate * ((layer.Vel_W / (1 - self.beta1**adam_t))
                                                    / (self.epsilon + np.sqrt(layer.Sqr_W / (1 - self.beta2**adam_t))))
                        layer.B -= learning_rate * ((layer.Vel_b / (1 - self.beta1**adam_t))
                                                    / (self.epsilon + np.sqrt(layer.Sqr_b / (1 - self.beta2**adam_t))))
                    else:
                        layer.W -= learning_rate * layer.dW
                        layer.B -= learning_rate * layer.dB

                self.loss_curve.append(self.loss)
            if epoch % 1 == 0:
                print(f"Loss after {epoch}s epoch: ", self.loss)

    def propagate(self, x, y):
        self.full_forward_propagation(x, y)
        self.full_backward_propagation(x, y)

    def forward_propagation(self, x):
        for lay in range(self.L + 1):
            if lay == 0:
                A_prev = x
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

    def full_forward_propagation(self, x, y):
        self.forward_propagation(x)
        self.compute_loss(y)

    def compute_dAL(self, y):
        al = self.layers[-1].A
        return (1 / y.shape[1]) * (al - y) / (al - al**2)

    def backward_propagation(self, x, d_AL, m):
        da = d_AL
        for lay in reversed(range(self.L + 1)):
            if lay == 0:
                prev = x
            else:
                prev = self.layers[lay - 1].A

            da = self.layers[lay].backward_propagate(da, prev, m, self.alpha)

    def full_backward_propagation(self, x, y):
        m = y.shape[0]
        da = self.compute_dAL(y)

        self.backward_propagation(x, da, m)

    def predict(self, inputs):
        return self.forward_propagation(inputs)
