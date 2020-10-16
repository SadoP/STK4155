import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_outputs, name, learning_rate=0.1):
        self.biases = 0.01*np.ones((n_outputs, 1))
        self.weights = np.random.rand(n_outputs, n_inputs)
        self.output = np.zeros((n_outputs, 1))
        self.z = np.array([])
        self.x = np.array([])
        self.error = np.array([])
        self.learning_rate = 0.01
        self.name = name

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.weights, self.x) + self.biases
        self.output = self.activation(self.z)

    @staticmethod
    def activation(x):
        return ActivationFunctions.sigmoid(x)

    @staticmethod
    def activation_grad(x) -> np.array:
        return ActivationFunctions.sigmoid_grad(x)

    def set_error(self, error) -> np.array:
        self.error = error

    def backward(self):
        pass

    def weight_grad(self):
        return np.matmul(self.error, self.x.T)

    def bias_grad(self) -> np.array:
        return np.sum(self.error, axis=0)

    def back_propagation(self):
        next_error = np.matmul(self.weights.T, self.error) * self.activation_grad(self.x)
        self.biases = self.biases + self.learning_rate * self.bias_grad()
        self.weights = self.weights + self.learning_rate * self.weight_grad()
        return next_error


class SoftMax:
    def __init__(self, n_inputs, n_outputs):
        self.biases = 0.01*np.ones((n_outputs, 1))
        self.weights = np.random.rand(n_outputs, n_inputs)
        self.output = np.zeros((n_outputs, 1))
        self.z = np.array([])
        self.x = np.array([])
        self.error = np.array([])
        self.learning_rate = 0.01

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.weights, self.x) + self.biases
        self.output = self.activation(self.z)

    @staticmethod
    def activation(x):
        return ActivationFunctions.probability(x)

    @staticmethod
    def activation_grad(x) -> np.array:
        return ActivationFunctions.probability_grad(x)

    def set_error(self, error) -> np.array:
        self.error = error

    def weight_grad(self):
        return np.matmul(self.error, self.x.T)

    def bias_grad(self) -> np.array:
        return np.sum(self.error, axis=0)

    def back_propagation(self):
        next_error = np.matmul(self.weights.T, self.error) * self.activation_grad(self.x)
        self.biases = self.biases + self.learning_rate * self.bias_grad()
        self.weights = self.weights + self.learning_rate * self.weight_grad()
        return next_error


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(x):
        return ActivationFunctions.sigmoid(x)*(1-ActivationFunctions.sigmoid(x))

    @staticmethod
    def probability(x):
        z = np.exp(x)
        return z / np.sum(z, axis=0, keepdims=True)

    @staticmethod
    def probability_grad(x):
        z = np.exp(x)
        a = np.sum(z)
        return x * (a-x)/a


RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
n_inputs = 5
n_middle1 = 64
n_middle2 = 128
n_outputs = 5
n_samples = 5
#input = np.array([[-5, 2, 3, 7, 0], [6, 10, -2, -5, -3]])
#y = np.array([[1,2,3,4,5]]).T
input = np.random.rand(n_samples, n_inputs).T
y = np.random.rand(n_samples, n_inputs)
y = y / np.sum(y, axis=1).reshape(n_samples, 1)

ld1 = LayerDense(n_inputs, n_middle1, "ld1")
ld2 = LayerDense(n_middle1, n_middle2, "ld2")
ls = SoftMax(n_middle2, n_outputs)
err = []
out = []
wei = []
bia = []
C = []

for i in range(200):
    ld1.forward(input)
    ld2.forward(ld1.output)
    ls.forward(ld2.output)
    error = y.T - ls.output
    ls.set_error(error)
    ld2.set_error(ls.back_propagation())
    ld1.set_error(ld2.back_propagation())
    ld1.back_propagation()
    err.append(ls.error)
    out.append(ls.output)
    C.append(np.sum(error**2))

