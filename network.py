import numpy as np
from typing import List
from console_progressbar import ProgressBar

RANDOM_SEED = 1337


class Metrics:
    @staticmethod
    def pred_to_class(y):
        r = np.argmax(y, axis=0)
        return r

    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = Metrics.pred_to_class(y_true)
        y_pred = Metrics.pred_to_class(y_pred)
        return np.sum(y_true == y_pred) / y_pred.__len__()

    @staticmethod
    def mse(y_true, y_pred):
        return np.sum(Costfunctions.mse(y_true, y_pred, None))

    @staticmethod
    def ce(y_true, y_pred):
        return np.sum(Costfunctions.cross_entropy(y_true, y_pred, None))

    @staticmethod
    def ce_grad(y_true, y_pred):
        return np.max(np.abs(Costfunctions.cross_entropy_grad(y_true, y_pred, None)))

    @staticmethod
    def coeff_determination(y_true, y_pred):
        r = (np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_pred)))
        return np.sum(1 - np.sum(r, axis=1))

class Costfunctions:
    la = 0.1
    eps = 1e-15

    @staticmethod
    def mse(y_true, y_pred, l):
        return 1/2*sum((y_true - y_pred)**2)

    @staticmethod
    def mse_grad(y_true, y_pred, l):
        return y_true - y_pred

    @staticmethod
    def ridge(y_true, y_pred, l):
        return np.sum((y_true - y_pred)**2) + Costfunctions.la*np.sum(l**2)

    @staticmethod
    def ridge_grad(y_true, y_pred, l):
        return y_true - y_pred + Costfunctions.la*l

    @staticmethod
    def cross_entropy(y_true, y_pred, l):
        y_pred = y_pred + Costfunctions.eps
        return -np.nansum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred), axis=0)

    @staticmethod
    def cross_entropy_grad(y_true, y_pred, l):
        y_pred = y_pred+Costfunctions.eps
        return - (y_true - y_true/y_pred + ((y_true-1)+(1-y_true)/(1-y_pred)))

    @staticmethod
    def soft_max(y_true, y_pred, l):
        z = np.exp(y_pred)
        mul = np.log((z.T/np.sum(z, axis=1)).T)
        res = - np.sum(y_true * mul, axis=1)
        return res

    @staticmethod
    def soft_max_grad(y_true, y_pred, l):
        z = np.exp(y_pred)
        res = y_true * (1/z - (z.T/np.sum(z, axis=1)).T)
        return res


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

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_grad(x):
        return 1


class Layer:
    def __init__(self, n_outputs, name, learning_rate, cf, af):
        self.output = np.zeros((n_outputs, 1))
        self.z = np.array([])
        self.x = np.array([])
        self.error = np.array([])
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.beta = 0.9
        self.weight_s = 0
        self.bias_s = 0
        self.eps = 1e-8
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

    def cost_function(self, y, yt, l):
        return self.cf(y, yt, l)

    def cost_grad(self, y, yt, l):
        return self.cf_grad(y, yt, l)

    def increment_epoch(self):
        self.epoch += 1


class LayerDense(Layer):
    def __init__(self, n_inputs, n_outputs, name, learning_rate, cost, activation):
        Layer.__init__(self, n_outputs, name, learning_rate, cost, activation)
        self.biases = 0.1*np.ones((n_outputs, 1))
        self.weights = 1*np.random.randn(n_outputs, n_inputs)
        self.bg = []
        self.wg = []
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
        b_grad = self.bias_grad()
        w_grad = self.weight_grad()
        self.bias_s = self.bias_s * self.beta + (1-self.beta)*b_grad**2
        self.weight_s = self.weight_s * self.beta + (1-self.beta)*w_grad**2

        self.bg = b_grad / np.sqrt(self.bias_s + self.eps)
        self.wg = w_grad / np.sqrt(self.weight_s + self.eps)

        self.biases = self.biases + self.learning_rate * self.bg
        self.weights = self.weights + self.learning_rate * self.wg
        return next_error


class Network:
    def __init__(self, layers: List[Layer], name: str, mf: []):
        self.layers = layers
        self.train_M = None
        self.test_M = None
        self.name = name
        self.mf = mf
        np.random.seed(RANDOM_SEED)

    def train(self, x, y, epochs, batch_size, x_test, y_test):
        batches = int(np.floor(x.shape[1] / batch_size))
        p = 0
        pb = ProgressBar(total=epochs, prefix='', suffix='', decimals=3,
                         length=50, fill='=',
                         zfill='>')
        self.app_metrics(x, x_test, y, y_test)
        pb.print_progress_bar(p)
        for i in range(epochs):
            for j in range(batches):
                xn = x[:, j * batch_size:(j + 1) * batch_size]
                yn = y[:, j * batch_size:(j + 1) * batch_size]
                self.forward_pass(xn)
                l = (np.sum(np.abs(self.layers[-1].weights), axis=1) + np.abs(self.layers[-1].biases.T)).T
                self.backward_pass(self.layers[-1].cost_grad(yn, self.layers[-1].output, l))
                lr = self.layers[-1].initial_learning_rate / (i*batches+j+1)
            self.app_metrics(x, x_test, y, y_test)
            self.inc_epoch()
            p += 1
            pb.print_progress_bar(p)

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
        for layer in self.layers:
            layer.learning_rate = lr

    def metric(self, y_true, y_pred):
        ms = []
        if self.mf is None:
            return ms
        for m in self.mf:
            ms.append(m(y_true, y_pred))
        return np.expand_dims(np.array(ms).T, axis=1)

    def app_metrics(self, x_train, x_test, y_train, y_test):
        if self.mf is None:
            return
        if self.train_M is None:
            self.train_M = self.metric(y_train, self.predict(x_train))
        else:
            self.train_M = np.append(self.train_M, self.metric(y_train, self.predict(x_train)),
                                     axis=1)
        if self.test_M is None:
            self.test_M = self.metric(y_test, self.predict(x_test))
        else:
            self.test_M = np.append(self.test_M, self.metric(y_test, self.predict(x_test)), axis=1)

    def get_train_met(self):
        return self.train_M.T

    def get_test_met(self):
        return self.test_M.T
