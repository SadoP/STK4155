import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from typing import List


class Costfunctions:
    @staticmethod
    def mse(y, yt):
        return sum((y - yt)**2)

    @staticmethod
    def mse_grad(y, yt):
        return y - yt

    @staticmethod
    def cross_entropy(y, yhat):
        """
        Calculates cross entropy loss
        :param y: true result
        :param yhat: calculated / predicted result
        :return:
        """
        return -np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    @staticmethod
    def cross_entropy_grad(y, yhat):
        """
        Calculates derivative of cross entropy loss
        :param y: true result
        :param yhat: calculated / predicted result
        :return:
        """
        return (yhat - y)/(yhat - yhat**2)


class ActivationFunctions:
    alpha = 0.02

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(x):
        return ActivationFunctions.sigmoid(x)*(1-ActivationFunctions.sigmoid(x))

    @staticmethod
    def softmax(x):
        z = np.exp(x)
        return z / np.sum(z, axis=0, keepdims=True)

    @staticmethod
    def softmax_grad(x):
        z = np.exp(x)
        a = np.sum(z)
        return x * (a-x)/a

    @staticmethod
    def relu(x):
        return (x > 0)*x

    @staticmethod
    def relu_grad(x):
        return (x > 0)*1

    @staticmethod
    def elu(x):
        return (x > 0) * x + (x < 0) * ActivationFunctions.alpha * (np.exp(x)-1)

    @staticmethod
    def elu_grad(x):
        return (x > 0) * 1 + (x < 0) * np.exp(x)*ActivationFunctions.alpha

    @staticmethod
    def leaky_relu(x):
        return (x > 0) * x + (x < 0) * x * ActivationFunctions.alpha

    @staticmethod
    def leaky_relu_grad(x):
        return (x > 0) * 1 + (x < 0) * ActivationFunctions.alpha


class Layer:
    def __init__(self, n_outputs, name, learning_rate, cf, af):
        self.output = np.zeros((n_outputs, 1))
        self.z = np.array([])
        self.x = np.array([])
        self.error = np.array([])
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.name = name
        self.epoch = 0
        self.cf = cf
        self.cf_grad = getattr(Costfunctions, cf.__name__+"_grad")
        self.af = af
        self.af_grad = getattr(ActivationFunctions, af.__name__+"_grad")

    def forward(self, x):
        pass

    def backwards(self, error):
        pass

    def activation(self, x):
        return self.af(x)

    def activation_grad(self, x):
        return self.af_grad(x)

    def cost_function(self, y, yt):
        return self.cf(y, yt)

    def cost_grad(self, y, yt):
        return self.cf_grad(y, yt)

    def increment_epoch(self):
        self.epoch += 1


class LayerDense(Layer):
    def __init__(self,n_inputs, n_outputs, name, learning_rate, cost, activation):
        Layer.__init__(self, n_outputs, name, learning_rate, cost, activation)
        self.biases = 0.01*np.ones((n_outputs, 1))
        self.weights = np.random.rand(n_outputs, n_inputs)
        self.initial_weights = self.weights
        self.initial_biases = self.biases

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.weights, self.x) + self.biases
        self.output = self.activation(self.z)

    def weight_grad(self):
        return np.matmul(self.error, self.x.T)

    def bias_grad(self):
        return np.expand_dims(np.sum(self.error, axis=1), axis=1)

    def backwards(self, error):
        self.error = error
        next_error = np.matmul(self.weights.T, self.error) * self.activation_grad(self.x)
        self.biases = self.biases + self.learning_rate * self.bias_grad()
        self.weights = self.weights + self.learning_rate * self.weight_grad()
        return next_error


class Network:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.C = []

    def train(self, x, y, epochs, batch_size):
        self.C = []
        batches = int(np.floor(x.shape[1] / batch_size))
        for i in range(epochs):
            for j in range(batches):
                xn = x[:, j * batch_size:(j + 1) * batch_size]
                yn = y[:, j * batch_size:(j + 1) * batch_size]
                self.forward_pass(xn)
                error = layers[-1].cost_function(yn, self.layers[-1].output)
                self.backward_pass(self.layers[-1].cost_grad(yn, self.layers[-1].output))
            self.C.append(np.sum(error ** 2))
            network.inc_epoch()

    def predict(self, x):
        return self.forward_pass(x)

    def forward_pass(self, x):
        for layer in self.layers:
            layer.forward(x)
            x = layer.output
        return x

    def backward_pass(self, error):
        for layer in reversed(self.layers):
            error = layer.backwards(error)

    def inc_epoch(self):
        for layer in self.layers:
            layer.increment_epoch()

    def adapt_learning_rate(self, lr):
        print("new learning rate" + str(lr))
        for layer in self.layers:
            layer.learning_rate = lr


def test_fun(x):
    return np.expand_dims(np.sum(x, axis=1), 1)

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
n_inputs = 5
n_middle1 = 64
n_middle2 = 1
n_outputs = 1
n_samples = 1000
batch_size = 100
split_size = 0.8
epochs = 250




input = np.random.rand(n_samples, n_inputs)
y = test_fun(input)
#input = scale(input, axis=1)

x_train, x_test, y_train, y_test = train_test_split(input, y, train_size=split_size)


ld1 = LayerDense(n_inputs, n_middle1, "ld1", 0.00001, Costfunctions.mse, ActivationFunctions.sigmoid)
ld2 = LayerDense(n_middle1, n_middle2, "ld2", 0.00001, Costfunctions.mse, ActivationFunctions.relu)
#ls = LayerDenseSoftMax(n_middle2, n_outputs, "ls1", 0.001)
layers = [ld1, ld2]
network = Network(layers)

network.train(x_train.T, y_train.T, epochs, batch_size)
print(network.C)


print(np.sum((network.predict(x_test.T) - y_test.T)**2))
#print(network.layers[-1].error)

