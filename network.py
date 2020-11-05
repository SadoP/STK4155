import numpy as np
from typing import List
from console_progressbar import ProgressBar

RANDOM_SEED = 1337


class Metrics:
    """
    Class to provide different functions that will be accepted by the neural network as a metric.
    Some of the functions just forward their input to their corresponding Costfunctions.
    Accuracy is only available as a metric, not as a cost function.
    """
    @staticmethod
    def __pred_to_class(y):
        """
        Returns the predicted class as integer, based on a vector of probabilities.
        :param y: probability vector
        :return: class with highest probability
        """
        r = np.argmax(y, axis=0)
        return r

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates the accuracy of the prediction vs the true result.
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: probability of agreement
        """
        y_true = Metrics.__pred_to_class(y_true)
        y_pred = Metrics.__pred_to_class(y_pred)
        return np.sum(y_true == y_pred) / len(y_pred)

    @staticmethod
    def mse(y_true, y_pred):
        """
        Wrapper for mean squared error cost function. The cost function accepts batches with n>1,
        the evaluated metric is therefore the mean over the different mses. This is no different 
        than calculating the mse over one large batch where all the data is somehow conglomerated.
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: Mean squared error metric
        """
        return np.sum(Costfunctions.mse(y_true, y_pred, None)) / y_pred.shape[1]

    @staticmethod
    def ce(y_true, y_pred):
        """
        Wrapper for cross entropy. Just forwards to CE as costfunction and calculates mean over 
        batch.
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: Cross Entropy metric
        """
        return np.sum(Costfunctions.cross_entropy(y_true, y_pred, None)) / y_pred.shape[1]

    @staticmethod
    def ce_grad(y_true, y_pred):
        """
        Average value of the absolute value of the gradient of the cross entropy. Used mostly for
        debugging and to determin whether the nn has converged to a solution.
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: Cross entropy gradient metric
        """
        y_pred = y_pred + Costfunctions.eps
        return np.mean(np.abs((y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]))

    @staticmethod
    def coeff_determination(y_true, y_pred):
        """
        R^2 value but adapted to fit into my code
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: R^2 value metric
        """
        r = (np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - np.sum(r)


class Costfunctions:
    """
    Class that defines my Costfunctions that are accepted by the neural network as valid cost 
    functions. Every cost function needs a gradient associated with it. 
    """
    
    """la is the parameter for ridge regression"""
    la = 0.1
    """small value for cross entropy to avoid log(0) or division by zero"""
    eps = 1e-15

    @staticmethod
    def mse(y_true, y_pred, l):
        """
        Mean squared error
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases. Not used here
        :return: mean squared error per batch and node
        """
        return 1/2*np.sum((y_true - y_pred)**2, axis=1)

    @staticmethod
    def __mse_grad(y_true, y_pred, l):
        """
        Gradient of mean squared error
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases. Not used here
        :return: gradient of mse per batch and node
        """
        return y_pred - y_true

    @staticmethod
    def ridge(y_true, y_pred, l):
        """
        Ridge error based on the parameter la.
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases
        :return: ridge error per batch and node
        """
        return np.sum((y_true - y_pred)**2) + Costfunctions.la*np.sum(l**2)

    @staticmethod
    def __ridge_grad(y_true, y_pred, l):
        """
        Gradient of ridge error
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases
        :return: ridge error gradient per batch and node
        """
        return -(y_true - y_pred + Costfunctions.la*l)

    @staticmethod
    def cross_entropy(y_true, y_pred, l):
        """
        Cross entropy error
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases. Not used here.
        :return: cross entropy error per batch and node
        """
        y_pred = y_pred + Costfunctions.eps
        return - np.nanmean(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred), axis=0)

    @staticmethod
    def __cross_entropy_grad(y_true, y_pred, l):
        """
        Gradient of cross entropy
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :param l: sum of values of weights and biases. Not used here.
        :return: gradient of cross entropy per batch and node
        """
        y_pred = y_pred + Costfunctions.eps
        return (y_pred - y_true)/(y_pred*(1-y_pred))/y_true.shape[0]


class ActivationFunctions:
    """
    Class that defines the activation functions that are accepted by the layers my NN consists 
    out of. Each activation function needs an accompanying gradient function.
    """
    
    """alpha is the small value for leaky relu"""
    alpha = 0.02
    """beta is the value for exponential linear unit"""
    beta = 1

    @staticmethod
    def sigmoid(x):
        """
        The basic sigmoid activation function
        :param x: input
        :return: output
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_grad(x):
        """
        Gradient of the sigmoid function
        :param x: input
        :return: output
        """
        return ActivationFunctions.sigmoid(x)*(1-ActivationFunctions.sigmoid(x))

    @staticmethod
    def softmax(x):
        """
        The softmax function. The result is the probability of a class based on the input vector.
        :param x: input
        :return: output
        """
        z = np.exp(x)
        return z / np.sum(z, axis=0, keepdims=True)

    @staticmethod
    def __softmax_grad(x):
        """
        Gradient of the softmax function
        :param x: input
        :return: output
        """
        z = np.exp(x)
        a = np.sum(z)
        return x * (a-x)/a

    @staticmethod
    def relu(x):
        """
        The ReLU function
        :param x: input
        :return: output
        """
        return (x > 0)*x

    @staticmethod
    def __relu_grad(x):
        """
        Gradient of the ReLU function
        :param x: input
        :return: output
        """
        return (x > 0)*1

    @staticmethod
    def elu(x):
        """
        The ELU function
        :param x: input
        :return: output
        """
        return (x > 0) * x + (x < 0) * ActivationFunctions.beta * (np.exp(x)-1)

    @staticmethod
    def __elu_grad(x):
        """
        Gradient of the ELU function
        :param x: input
        :return: output
        """
        return (x > 0) * 1 + (x < 0) * np.exp(x)*ActivationFunctions.beta

    @staticmethod
    def leaky_relu(x):
        """
        The leaky ReLU function
        :param x: input
        :return: output
        """
        return (x > 0) * x + (x < 0) * x * ActivationFunctions.alpha

    @staticmethod
    def __leaky_relu_grad(x):
        """
        Gradient of the leaky ReLU function
        :param x: input
        :return: output
        """
        return (x > 0) * 1 + (x < 0) * ActivationFunctions.alpha

    @staticmethod
    def linear(x):
        """
        The linear output function
        :param x: input
        :return: output
        """
        return x

    @staticmethod
    def __linear_grad(x):
        """
        Gradient of the linear output function
        :param x: input
        :return: output
        """
        return 1


class Layer:
    """
    The basic layer class. Other layer classes can be abstracted from the basic layer class.
    It currently provides the sceleton for errors, learning rate, inputs and outputs, cost- and
    activation functions.
    """
    def __init__(self, n_outputs, name, learning_rate, cf, af):
        """
        The constructor for the base layer.
        :param n_outputs: Amount of neurons in this layer
        :param name: Name of the layer. Strictly speaking not necessary, but makes identification
            easier
        :param learning_rate: Learning rate of the layer
        :param cf: Costfunction of the layer
        :param af: Activationfunction of the layer
        """
        # Initializing the different variables
        # The final output of the layer
        self.output = np.zeros((n_outputs, 1))
        # Output before being passed through the activation function
        self.z = np.array([])
        # Input of the layer
        self.x = np.array([])
        # Error for each neuron, based on backpropagation
        self.error = np.array([])
        # Keeping track of the learning rate in case it is adapted
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        # variables for RMSProp
        self.beta = 0.9
        self.weight_s = 0
        self.bias_s = 0
        self.eps = 1e-8
        self.name = name
        # Setting cost and activation functions as well as their gradients
        self.cf = cf
        self.cf_grad = getattr(Costfunctions, "__"+cf.__name__+"_grad")
        self.af = af
        self.af_grad = getattr(ActivationFunctions, "__"+af.__name__+"_grad")

    """
    The Layer class is used to abstract the actual layers from. Depending on the layers, 
    forwards and backwards passes will look differently. Because Every layer will have those 
    functions, they are defined here, but each layer will overwrite them with their own.
    """

    def forward(self, x):
        pass

    def backwards(self, error):
        pass

    """
    The activation and cost functions are defined in the constructor. Every layer will have one 
    but since they are interchangeable and don't depend on the layer, they can be defined here.
    """
    def activation(self, x):
        return self.af(x)

    def activation_grad(self, x):
        return self.af_grad(x)

    def cost_function(self, y, yt, l):
        return self.cf(y, yt, l)

    def cost_grad(self, y, yt, l):
        return self.cf_grad(y, yt, l)


class LayerDense(Layer):
    """
    The class for a dense layer. It is abstracted from the basic layer class.
    """
    def __init__(self, n_inputs, n_outputs, name, learning_rate, cost, activation):
        """
        The constructor passes most of its arguments on to the constructor for the base class.
        :param n_inputs: Amount of inputs into the layer
        :param n_outputs: Amount of neurons in this layer
        :param name: Name of the layer. Strictly speaking not necessary, but makes identification
            easier
        :param learning_rate: Learning rate of the layer
        :param cost: Costfunction of the layer
        :param activation: Activationfunction of the layer
        """
        Layer.__init__(self, n_outputs, name, learning_rate, cost, activation)
        # initializing weights and biases for the dense layer.
        self.biases = 0.1*np.ones((n_outputs, 1))
        self.weights = 1*np.random.randn(n_outputs, n_inputs)
        # initializing for RMSProp
        self.bg = []
        self.wg = []
        # keeping track of weights and biases after initialization
        self.initial_weights = self.weights
        self.initial_biases = self.biases

    def forward(self, x) -> None:
        """
        Calculates the forwards pass for a dense layer. It does not return anything, but saves
        the result as the output of the layer.
        :param x: input
        :return: None
        """
        self.x = x
        self.z = np.dot(self.weights, self.x) + self.biases
        self.output = self.activation(self.z)

    def __weight_grad(self):
        """
        Calculates the gradient of the weights for backpropagation, based on the error returned
        from the previous layer and the values of its current inputs.
        :return: Gradient of the weights
        """
        return np.matmul(self.error, self.x.T)

    def __bias_grad(self):
        """
        Calculates the gradient of the biases for backpropagation, based on the error returned
        from the previous layer and the values of its current inputs.
        :return: Gradient of the biases
        """
        return np.expand_dims(np.sum(self.error, axis=1), axis=1)

    def backwards(self, error):
        """
        Calculates the backwards pass for the layer
        :param error: The error backpropagated from the previous layer or the cost function in
        case this is the output layer
        :return: The error that needs to be propagated backwards to the next layer
        """
        # saving the current error for this layer and calculating the backpropagated error
        self.error = error
        next_error = np.matmul(self.weights.T, self.error) * self.activation_grad(self.x)
        # calculating gradients for weights and biases
        b_grad = self.__bias_grad()
        w_grad = self.__weight_grad()

        # Applying RMSProp to update weights and biases
        self.bias_s = self.bias_s * self.beta + (1-self.beta)*b_grad**2
        self.weight_s = self.weight_s * self.beta + (1-self.beta)*w_grad**2

        self.bg = b_grad / np.sqrt(self.bias_s + self.eps)
        self.wg = w_grad / np.sqrt(self.weight_s + self.eps)

        self.biases = self.biases - self.learning_rate * self.bg
        self.weights = self.weights - self.learning_rate * self.wg
        return next_error


class Network:
    """
    The class that fully defines a neural network
    """
    def __init__(self, layers: List[Layer], name: str, mf: []):
        """
        The constructor for the neural network. It takes a list of already initialized layers as
        an input, where the first layer in the list is the input layer and the last layer is the
        output layer. The constructor does not perform any checking for errors, so it will just
        break if the amount of outputs of one layer are not equal to the amount of inputs for the
        next layer.
        Due to the way the Network is built, it will be possible to change layers, activation or
        cost functions, inspect values within the layers of the network or just continue training
        the network after a certain stage.
        The network also takes in a list of metric functions. Before training begins and after
        each epoch, the metric functions will be applied to training and testing data,
        to keep track of how the neural network performs.
        :param layers: The layers the neural network consists of.
        :param name: Name of the network
        :param mf: List of metric functions as defined in the above class
        """
        self.layers = layers
        self.train_M = None
        self.test_M = None
        self.name = name
        self.mf = mf
        # initializing epoch count
        self.epoch = 0
        np.random.seed(RANDOM_SEED)

    def __forward_pass(self, x):
        """
        A full forwards pass through the whole network. The outputs of the network are stored in
        each layer.
        :param x: input of the first layer
        :return: Output of the last layer
        """
        for layer in self.layers:
            layer.forward(x)
            x = layer.output
        return x

    def predict(self, x):
        """
        Public function that fulfills the same functionality as the forward pass
        :param x: input of the first layer
        :return: Output of the last layer
        """
        return self.__forward_pass(x)

    def __backward_pass(self, error):
        """
        Goes backwards through the layers and calls the backwards function for each layer,
        with the error calculated from the previous layer
        :param error: initial error for the output layer
        :return: None
        """
        for layer in reversed(self.layers):
            error = layer.backwards(error)

    def __inc_epoch(self):
        """
        Increases the epoch count
        :return: None
        """
        self.epoch += 1

    def train(self, x, y, epochs, batch_size, x_test, y_test):
        """
        Training the neural networ
        :param x: input vector of training data
        :param y: output vector of training data
        :param epochs: number of epochs to be trained
        :param batch_size: size of batches the training data is split into
        :param x_test: input vector of testing data
        :param y_test: output vector of testing data
        :return: None
        """
        batches = int(np.floor(x.shape[1] / batch_size))
        p = 0
        pb = ProgressBar(total=epochs, prefix='', suffix='', decimals=3,
                         length=50, fill='=',
                         zfill='>')
        # Calculating metrics for training and testing data before training the network
        self.__app_metrics(x, x_test, y, y_test)
        pb.print_progress_bar(p)
        # going through batches and epochs
        for i in range(epochs):
            for j in range(batches):
                xn = x[:, j * batch_size:(j + 1) * batch_size]
                yn = y[:, j * batch_size:(j + 1) * batch_size]
                self.__forward_pass(xn)
                l = (np.sum(np.abs(self.layers[-1].weights), axis=1) + np.abs(self.layers[-1].biases.T)).T
                self.__backward_pass(self.layers[-1].cost_grad(yn, self.layers[-1].output, l))
            # Calculating metrics are every epoch
            self.__app_metrics(x, x_test, y, y_test)
            self.__inc_epoch()
            p += 1
            pb.print_progress_bar(p)

    def adapt_learning_rate(self, lr):
        """
        Adapts the learning rate for every layer
        :param lr: new learning rate
        :return: None
        """
        for layer in self.layers:
            layer.learning_rate = lr

    def __metric(self, y_true, y_pred):
        """
        Calculates every metric in the previously defined list of metrics for the given data
        :param y_true: Data to test against
        :param y_pred: Predicted result
        :return: The result for every metric
        """
        ms = []
        if self.mf is None:
            return ms
        for m in self.mf:
            ms.append(m(y_true, y_pred))
        return np.expand_dims(np.array(ms).T, axis=1)

    def __app_metrics(self, x_train, x_test, y_train, y_test):
        """
        Calculates metrics for test and training data and appends to the preinitialized list
        :param x_train: training input
        :param x_test: testing input
        :param y_train: known true result for training data
        :param y_test: known true result for testing data
        :return:
        """
        if self.mf is None:
            return
        if self.train_M is None:
            self.train_M = self.__metric(y_train, self.predict(x_train))
        else:
            self.train_M = np.append(self.train_M, self.__metric(y_train, self.predict(x_train)),
                                     axis=1)
        if self.test_M is None:
            self.test_M = self.__metric(y_test, self.predict(x_test))
        else:
            self.test_M = np.append(self.test_M, self.__metric(y_test, self.predict(x_test)), axis=1)

    def get_train_met(self):
        """
        :return: metrics for training data
        """
        return self.train_M.T

    def get_test_met(self):
        """
        :return: metrics for testing data
        """
        return self.test_M.T
