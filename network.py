import sys

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
        return np.sum(y_true == y_pred) / y_true.__len__()

    @staticmethod
    def mse(y_true, y_pred):
        return Costfunctions.mse(y_true, y_pred, None)


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
        #print(y_true)
        #print(y_pred)
        #print(np.sum(-y_true*np.log(y_pred), axis=0))
        #print(y_true.shape)
        #print(y_pred.shape)
        #print(np.sum(-y_true*np.log(y_pred), axis=0).shape)
        #print(np.max(np.sum(-y_true*np.log(y_pred), axis=0)))
        y_pred = y_pred + Costfunctions.eps
        #print(y_true)
        #print(y_pred)
        #print(-np.sum(y_true*np.log(y_pred) +
        #               (1-y_true)*np.log(1-y_pred), axis=0))
        #print(y_pred.shape)
        #print("sum")
        #print(np.nansum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred), axis=0).shape)
        return -np.nansum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred), axis=0)

    @staticmethod
    def cross_entropy_grad(y_true, y_pred, l):
        #print(y_true.shape)
        #y_true = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 1]])
        #y_pred = np.array([[0, 0.4, 0.3, 0.1, 0], [0.2, 0.4, 0.4, 0.2, 0], [0.8, 0.2, 0.3, 0.7, 1]])
        #eps = 1e-15
        y_pred = y_pred+Costfunctions.eps
        #print(y_true)
        #print(y_pred)
        #print(y_true.shape)
        #print(y_pred.shape)
        #print((-y_true/y_pred).shape)
        #print(-y_true/y_pred)
        #print(np.max(np.abs(-y_true/y_pred + (1-y_true)/(1-y_pred))))
        #(2*y_true-1)*(1-1/y_pred)
        #print(- (y_true - y_true/y_pred + ((1-y_true)+(1-y_true)/(1-y_pred))))
        #sys.exit()
        #print(- (y_true - y_true/y_pred + ((1-y_true)+(1-y_true)/(1-y_pred))))
        return - (y_true - y_true/y_pred + ((1-y_true)+(1-y_true)/(1-y_pred)))

    @staticmethod
    def soft_max(y_true, y_pred, l):

        #y_true = np.array([[0,0,1],[0,1,0]])
        #y_pred = np.array([[0,0.8,0.2],[0.2,0.4,0.8]])
        #print(y_pred.shape)
        z = np.exp(y_pred)
        #print(z.shape)
        #print(y_true.shape)
        #print(np.sum(z, axis=0).shape)
        mul = np.log((z.T/np.sum(z, axis=1)).T)
        #print(mul.shape)
        #print(np.sum(z))
        #print(np.sum(mul))
        res = - np.sum(y_true * mul, axis=1)
        #print(res)
        return res

    @staticmethod
    def soft_max_grad(y_true, y_pred, l):
        #y_true = np.array([[0,0,1],[0,1,0]])
        #y_pred = np.array([[0,0.8,0.2],[0.2,0.4,0.8]])
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
        #print(x)
        #print("z")
        #print(z)
        #sys.exit()
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
    def __init__(self,n_inputs, n_outputs, name, learning_rate, cost, activation):
        Layer.__init__(self, n_outputs, name, learning_rate, cost, activation)
        self.biases = 0.1*np.ones((n_outputs, 1))
        self.weights = 1*np.random.randn(n_outputs, n_inputs)
        #self.weights = (self.weights.T / np.sum(self.weights, axis=1)).T
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


        #self.biases = self.biases + self.learning_rate * self.bias_grad()
        #self.weights = self.weights + self.learning_rate * self.weight_grad()
        return next_error


class Network:
    def __init__(self, layers: List[Layer], name: str, mf):
        self.layers = layers
        self.train_M = []
        self.test_M = []
        self.name = name
        self.mf = mf
        np.random.seed(RANDOM_SEED)

    def train(self, x, y, epochs, batch_size, x_test, y_test):
        #self.train_C = []
        batches = int(np.floor(x.shape[1] / batch_size))
        p = 0
        pb = ProgressBar(total=epochs, prefix='', suffix='', decimals=3,
                         length=50, fill='=',
                         zfill='>')
        #self.forward_pass(x)
        #l = (np.sum(np.abs(self.layers[-1].weights), axis=1) + np.abs(self.layers[-1].biases.T)).T
        #error = self.layers[-1].cost_function(y, self.layers[-1].output, l)
        #self.train_C.append(np.sum(error))
        #self.forward_pass(x_test)
        #error = self.layers[-1].cost_function(y_test, self.layers[-1].output, l)
        self.train_M.append(self.metric(y, self.predict(x)))
        self.test_M.append(self.metric(y_test, self.predict(x_test)))
        pb.print_progress_bar(p)
        for i in range(epochs):
            for j in range(batches):
                xn = x[:, j * batch_size:(j + 1) * batch_size]
                yn = y[:, j * batch_size:(j + 1) * batch_size]
                self.forward_pass(xn)
                l = (np.sum(np.abs(self.layers[-1].weights), axis=1) + np.abs(self.layers[-1].biases.T)).T
                #print(l)
                #error = self.layers[-1].cost_function(yn, self.layers[-1].output, l)
                #print(error)
                #sys.exit()
                #print(np.max(self.layers[-1].cost_grad(yn, self.layers[-1].output, l)))
                self.backward_pass(self.layers[-1].cost_grad(yn, self.layers[-1].output, l))
                lr = self.layers[-1].initial_learning_rate / (i*batches+j+1)
                #self.adapt_learning_rate(lr)
            self.train_M.append(self.metric(y, self.predict(x)))
            self.test_M.append(self.metric(y_test, self.predict(x_test)))
            #self.train_C.append(np.sum(error))
            #self.forward_pass(x_test)
            #error = self.layers[-1].cost_function(y_test, self.layers[-1].output, l)
            #self.test_C.append(np.sum(error))
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
        pass
        #for layer in self.layers:
        #    layer.learning_rate = lr

    def metric(self, y_true, y_pred):
        return self.mf(y_true, y_pred)