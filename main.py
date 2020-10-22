import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.preprocessing import scale
from sklearn.utils import shuffle

import colormaps as cmaps


def franke(x: np.array, y: np.array, noise_level: float) -> np.array:
    """
    Calculates the value of the franke function for x and y coordinates and adds random noise based
    on the noise level
    :param x: x-coordinate
    :param y: y-coordinate
    :param noise_level: noise level
    :return: Values of the franke function for the x-y pairs
    """
    if x.any() < 0 or x.any() > 1 or y.any() < 0 or y.any() > 1:
        # Breaks if the interval of x and y is outside [0,1]
        print("Franke function is only valid for x and y between 0 and 1.")
        return None
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    noise = np.random.normal(0, 1, term4.shape)
    return term1 + term2 + term3 + term4 + noise_level * noise


def r2_error(y_data: np.array, y_model: np.array) -> float:
    """
    Calculates R squared based on predicted and true data.
    """
    r = ((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
    return 1-np.sum(r, axis=0)


def mse_error(y_data: np.array, y_model: np.array) -> float:
    """
    Calculates mean squared error based in predicted and true data
    """
    r = (y_data - y_model) ** 2
    return np.sum(r, axis=0) / np.size(y_model, 0)


def coords_to_polynomial(x: np.array, y: np.array, p: int) -> np.array:
    """
    Calculates the feature matrix based on input coordinates x and y and the given degree of the
    polynomial
    :param x: x vector
    :param y: y vector
    :param p: degree of polynomial
    :return: feature matrix
    """
    if x.shape != y.shape:
        # Breaks if x and y are of different shapes
        print("mismatch between size of x and y vector")
        return None
    # Preparing x and y and pre-allocating the feature matrix X
    x = x.flatten()
    y = y.flatten()
    feat_matrix = np.zeros(shape=(len(x), max_mat_len(p)))
    k = 0
    for i in range(p + 1):
        for j in range(i + 1):
            feat_matrix[:, k] = x ** (i - j) * y ** j
            k += 1
    return feat_matrix


def solve_lin_equ(y: np.array, x: np.array, solver: str = "ols", epochs: int = 0, batches: int = 0,
                  la: float = 1, g0: float = 1e-3) -> Tuple[np.array, np.array]:
    """
    Solves linear equation of type y = x*beta. This can be done using ordinary least squares (OLS),
    ridge or lasso regression. For ridge and lasso, an additional parameter l for shrinkage /
    normalization needs to be provided.
    :param y: solution vector of linear problem
    :param x: feature vector or matrix of problem
    :param solver: solver. Can be ols, ridge or lasso
    :param la: shrinkage / normalization parameter for ridge or lasso
    :param g: learning rate factor gamma
    :return: parameter vector that solves the problem & variance of parameter vector
    """
    if epochs <= 0 or batches <= 0:
        print("epoch or batch data invalid")
        sys.exit(-1)
    beta = np.random.random(size=(x.shape[1], epochs))
    g = g0
    bet = 0.9
    s = 0
    eps = 1e-8
    bl = int(np.floor(y.shape[0]/batches))
    currb = beta[:, 0]
    for e in range(epochs):
        for b in range(batches):
            yb = y[b * bl:(b + 1) * bl - 1]
            xb = x[b * bl:(b + 1) * bl - 1]
            c = cost_function(yb, xb, currb, solver)
            gd = cost_grad(yb, xb, currb, solver)
            s = bet*s + (1-bet)*gd**2
            #print(np.max((g * gd/(s+eps))/beta))
            currb = currb - g * gd/(np.sqrt(s+eps))
        beta[:, e] = currb
    return beta


def cost_function(y: np.array, x: np.array, beta: np.array, solver: str = "ols", la: float = 1) -> float:
    """
    Calculates cost function for ordinary least squares or ridge regression
    :param y: desired output
    :param x: feature matrix
    :param beta: parameters
    :param solver: solver. Can be ols or ridge
    :param la: regression parameter for ridge regression
    :return: Cost value
    """
    if solver == "ols":
        return 1/2*np.sum((y - x@beta)**2)
    elif solver == "ridge":
        return 1/2*np.sum((y - x@beta)**2) + 1/2*la*np.sum(beta**2)
    elif solver == "logistic":
        return -np.sum(y*np.log(x@beta) + (1-y)*np.log(1-x@beta))


def cost_grad(y: np.array, x: np.array, beta: np.array, solver: str = "ols", la: float = 1) -> np.array:
    """
    Calculates gradient of cost function in parameter direction
    :param y: desired output
    :param x: feature matrix
    :param beta: parameters
    :param solver: solver. Can be ols or ridge
    :param la: regression parameter for ridge regression
    :return: gradient of cost function
    """
    if solver == "ols":
        return - (y - x@beta)@x
    elif solver == "ridge":
        res = - (y - x@beta)@x + la * np.sum(beta)
        if np.isnan(np.sum(res)):
            print(y)
            print(x)
            print(beta)
            sys.exit()
        return res
    elif solver == "logistic":
        return (x@beta - y)/(x@beta - x@beta**2)


def create_grid(res: float) -> Tuple[np.array, np.array]:
    """
    Creates a parameter grid for x and y in [0,1] with a given resolution
    :param res: resolution of the grid
    :return: Parameter grid
    """
    x = np.arange(0, 1, res)
    y = np.arange(0, 1, res)
    x, y = np.meshgrid(x, y)
    return x, y


def create_data(deg: int, x: np.array = None, y: np.array = None, z: np.array = None) \
        -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Creates the data that is used for the regression. Depending on the given degree of the
    polynomial a feature matrix will be created. Also data for the franke function for this feature
    matrix is calculated. If cross validation is to be performed, the parameter with a value larger
    than one has to be set. For a value equal 1, the data is split into 80:20 training and test
    data. For cross validation, the data will be split into equal folds depending on the number
    given.
    If values for x,y and z are provided, no feature matrix is created but instead the given values
    are passed on. The function also performs scaling of the data, if necessary.
    :param deg: Degree of the polynomial
    :param x: x-vector
    :param y: y-vector
    :param z: z-vector
    :return: arrays containing feature and result vectors split into training and test
    """
    # creating coordinate vectors if none were provided
    if x is None or y is None:
        x, y = create_grid(RESOLUTION)
        z = franke(x, y, NOISE_LEVEL).flatten()
    # calculating feature matrix
    feature_mat = coords_to_polynomial(x, y, deg)
    # scaling
    if SCALE_DATA:
        feature_mat = scale(feature_mat, axis=1)
    # calculation z-vector if none was provided
    if z is None:
        z = franke(x, y, NOISE_LEVEL).flatten()
    # dimensions of the feature matrix
    n = int(feature_mat.shape[0])
    # test_l is the length of the test split
    # 80:20 split in train and test data
    test_l = int(np.floor(n * 0.2))

    # randomizing order of feature matrix and result vector and allocating split variables
    feature_mat, z = shuffle(feature_mat, z, random_state=RANDOM_SEED)
    feature_mat_train = np.delete(feature_mat, np.arange(0, test_l, 1, int), axis=0)
    feature_mat_test = feature_mat[:test_l, :]
    z_train = np.delete(z, np.arange(0, test_l, 1, int), axis=0)
    z_test = z[:test_l]
    z_test = np.expand_dims(z_test, 1)
    z_train = np.expand_dims(z_train, 1)
    # reshaping to make it easier in the coming functions. The first two indeces are left in their
    # previous relative and are the length and height of the features. The now last index represents
    # the fold.
    return feature_mat_train, feature_mat_test, z_train, z_test


def predict(beta: np.array, feature_mat: np.array) -> np.array:
    """
    Predicts result of linear equation: y = x*beta
    :param beta: parameter matrix
    :param feature_mat: feature matrix
    :return:
    """
    return feature_mat @ beta


def err_from_var(var: np.array, sample_size: int) -> np.array:
    """
    Calculates the error from its variance
    :param var: variance
    :param sample_size: Sample size
    :return: error based on variance and sample size
    """
    return 1.97 * np.sqrt(var) / np.sqrt(sample_size)


def train(deg: int, solver: str = "ols", la: float = 1, epochs: int = 0, batches: int = 0,
          x: np.array = None, y: np.array = None, z: np.array = None, g0: float = 1e-3) \
          -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculates fit based on given degree. If no data for x,y and z are given, data will be
    created from / for the franke function, otherwise the provided data will be used
    :param deg: Degree of the polynomial to be fitted onto
    :param solver: Algorithm for solving. Can be ols, ridge or lasso
    :param la: parameter for ridge and lasso regression
    :param x: x-vector
    :param y: y-vector
    :param z: z-vector
    :return: Regression parameter vector, error and variance of parameter vector, test & training
        data R squqared value and mean squared error
    """
    # creates feature matrix and result vector
    train_x, test_x, train_z, test_z = create_data(deg, x=x, y=y, z=z)
    # declares variables for results
    # solving the linear equation, calculating error for parameter vector
    beta = solve_lin_equ(train_z.flatten(), train_x, solver=solver, epochs=epochs, batches=batches,
                         la=la, g0=g0)
    # Calculate errors based on the chosen samples
    test_r = r2_error(test_z, predict(beta, test_x))
    test_m = mse_error(test_z, predict(beta, test_x))
    train_r = r2_error(train_z, predict(beta, train_x))
    train_m = mse_error(train_z, predict(beta, train_x))
    #print(test_r)
    return test_r, train_r, test_m, train_m, beta


def max_mat_len(maxdeg: int) -> int:
    """
    Calculates the size of the polyonmial based on its degree
    :param maxdeg: degree of polynomial
    :return: size of the polynomial
    """
    return int((maxdeg + 2) * (maxdeg + 1) / 2)


def train_degs(maxdeg: int, solver: str = "ols",
               la: float = 1, epochs: int = 0, batches: int = 0,
               x: np.array = None, y: np.array = None, z: np.array = None, g0: float = 1e-3) \
        -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    This is a wrapper for the function "train", that loops over all the degrees given in maxdeg
    :param maxdeg: highest order degree to be trained
    :param solver: Algorithm for solving. Can be ols, ridge or lasso
    :param la: parameter for ridge and lasso regression
    :param x: x-vector
    :param y: y-vector
    :param z: z-vector
    :return: Regression parameter vector, error and variance of parameter vector, test & training
        data R squqared value and mean squared error
    """
    # declare variables so that the values can be assigned in the loop
    beta = np.zeros(shape=(maxdeg, max_mat_len(maxdeg), epochs))
    var = np.zeros(shape=(maxdeg, max_mat_len(maxdeg)))
    err = np.zeros(shape=(maxdeg, max_mat_len(maxdeg)))
    test_r = np.zeros(shape=(maxdeg, epochs))
    train_r = np.zeros(shape=(maxdeg, epochs))
    test_m = np.zeros(shape=(maxdeg, epochs))
    train_m = np.zeros(shape=(maxdeg, epochs))
    # looping over every degree to be trained
    for i in range(maxdeg):
        print("training degree: " + str(i + 1))
        t = train(i + 1, solver=solver, epochs=epochs, batches=batches, la=la, x=x, y=y, z=z, g0=g0)
        b = t[4]
        beta[i, 0:len(b)] = b
        test_r[i, :] = t[0]
        train_r[i, :] = t[1]
        test_m[i, :] = t[2]
        train_m[i, :] = t[3]
    return test_r, train_r, test_m, train_m, beta


def print_errors(x_values: np.array, errors: np.array, labels: list, name: str, logy: bool = False,
                 logx: bool = False,
                 xlabel: str = "Degree", ylabel: str = "error value", d: int = 6) -> Figure:
    """
    Helper function to create similar looking graphs. All the graphs where mean squared errors or
    the r_squared value are plotted and shown in the report are plotted using this function
    :param x_values: values for x axis
    :param errors: values for y axis
    :param labels: plot labels
    :param name: filename
    :param logy: plot y logarithmically?
    :param logx: plot x logarithmically?
    :param xlabel: label for x
    :param ylabel: label for y
    :param d: size of plot
    :return: the created figure
    """
    # creating new figure
    fig = plt.figure(figsize=(d, d), dpi=300)
    # plotting every given value
    for i in range(len(errors)):
        label = labels[i]
        # linestyle depends on test or train data
        if label.__contains__("train"):
            linestyle = "--"
        else:
            linestyle = "-"
        plt.plot(x_values, errors[i], label=labels[i], linestyle=linestyle)
    plt.legend()
    ax = fig.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
#    fig.tight_layout()
    fig.savefig("images/" + name + ".png", dpi=300)
    return fig


def print_cont(deg: int, data: np.array, name: str, zlabel: str) -> Figure:
    """
    Similar as above but for contour data where the axes depend on the degree of the polynome
    :param deg: degree of the polynomial
    :param data: z-values
    :param name: filename
    :param zlabel: label for z axis
    :return: the create figure
    """
    # x axis is the parameter index
    x = np.linspace(1, max_mat_len(deg), max_mat_len(deg))
    # axis is the polynomial index
    y = np.linspace(1, deg, deg)
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, data, cmap=cmaps.parula)
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Polynomial Degree")
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    fig.savefig("images/" + name + ".png", dpi=300)
    return fig


def read_data_file() -> pd.DataFrame:
    """
    Reads file with own data into pandas data frame
    :return: File content as dataframe
    """
    data_file = DATA_FILE
    return pd.read_csv(data_file, delimiter=",")


def get_data() -> Tuple[np.array, np.array, np.array]:
    """
    I used some data that I had left over from my previous position. This is calculated magnetic
    field data from some permanent magnet configuration. The data does not follow a strictly
    integer-polynomial trend but can be described not too badly by a polynom within the boundaries.
    I made the data more sparse by randomly removing 80% of the original data.
    :return: prepared data
    """
    data = read_data_file()
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["B"])
    cutoff = int(np.floor(len(z) * .2))
    x, y, z = shuffle(x, y, z, random_state=RANDOM_SEED)
    x = x[:cutoff]
    y = y[:cutoff]
    z = z[:cutoff]
    z = z / np.max(z)
    return x, y, z


def print_data() -> Figure:
    """
    Prints my data to image file
    :return: The created figure
    """
    x, y, z = get_data()
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, c='y', marker='o')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("B")
    fig.savefig("images/data.png", dpi=300)
    return fig


def loop_over_batches(deg: int, values: np.array,
                             solver: str = "ridge", x: np.array = None, y: np.array = None,
                             z: np.array = None, la: float = 1, epochs: int = 0, g0: float = 1e-3):
    n = values.shape[0]
    test_r = np.zeros(shape=(deg, n, epochs))
    train_r = np.zeros(shape=(deg, n, epochs))
    test_m = np.zeros(shape=(deg, n, epochs))
    train_m = np.zeros(shape=(deg, n, epochs))
    beta = np.zeros(shape=(deg, max_mat_len(deg), n, epochs))
    i = 0
    for v in values:
        test_r[:, i, :], train_r[:, i, :], test_m[:, i, :], train_m[:, i, :], beta[:, :, i, :] = train_degs(
            maxdeg=deg, epochs=epochs, batches=v,
            solver=solver, la=la, x=x, y=y, z=z, g0=g0)
        i = i + 1
    return test_r[:, :, -1], train_r[:, :, -1], test_m[:, :, -1], train_m[:, :, -1], beta[:, :, :, -1]


def loop_over_epochs(deg: int, values: np.array,
                             solver: str = "ridge", x: np.array = None, y: np.array = None,
                             z: np.array = None, la: float = 1, batches: int = 0, g0:float = 1e-3):
    v = np.max(values)
    test_r, train_r, test_m, train_m, beta = train_degs(maxdeg=deg, epochs=v, batches=batches,
                                                        solver=solver, la=la, x=x, y=y, z=z, g0=g0)
    return test_r, train_r, test_m, train_m, beta


def loop_over_lambda(deg: int, values: np.array,
                             solver: str = "ridge", x: np.array = None, y: np.array = None,
                             z: np.array = None, epochs: int = 0, batches: int = 0, g0:float = 1e-3):
    n = values.shape[0]
    test_r = np.zeros(shape=(deg, n, epochs))
    train_r = np.zeros(shape=(deg, n, epochs))
    test_m = np.zeros(shape=(deg, n, epochs))
    train_m = np.zeros(shape=(deg, n, epochs))
    beta = np.zeros(shape=(deg, max_mat_len(deg), n, epochs))
    i = 0
    for v in values:
        test_r[:, i, :], train_r[:, i, :], test_m[:, i, :], train_m[:, i, :], beta[:, :, i, :] = train_degs(
            maxdeg=deg, epochs=epochs, batches=batches,
            solver=solver, la=v, x=x,
            y=y, z=z, g0=g0)
        i = i + 1
    return test_r[:, :, -1], train_r[:, :, -1], test_m[:, :, -1], train_m[:, :, -1], beta[:, :, :, -1]


def loop_over_learning_rate(deg: int, values: np.array,
                           solver: str = "ridge", x: np.array = None, y: np.array = None,
                           z: np.array = None, epochs: int = 0, batches: int = 0, la: float = 1):
    n = values.shape[0]
    test_r = np.zeros(shape=(deg, n, epochs))
    train_r = np.zeros(shape=(deg, n, epochs))
    test_m = np.zeros(shape=(deg, n, epochs))
    train_m = np.zeros(shape=(deg, n, epochs))
    beta = np.zeros(shape=(deg, max_mat_len(deg), n, epochs))
    i = 0
    for v in values:
        print(v)
        test_r[:, i, :], train_r[:, i, :], test_m[:, i, :], train_m[:, i, :], beta[:, :, i, :] = train_degs(
            maxdeg=deg, epochs=epochs, batches=batches, solver=solver, la=la, x=x, y=y, z=z, g0=v)
        i = i + 1
    return test_r[:, :, -1], train_r[:, :, -1], test_m[:, :, -1], train_m[:, :, -1], beta[:, :, :, -1]


def train_print_hyperparameter(deg: int, parameter: str, values: np.array,
                               solver: str = "ridge", x: np.array = None, y: np.array = None,
                               z: np.array = None, epochs: int = 0, batches: int = 0, la: float = 1,
                               g0: float = 1e-3, task: str = "a", d: int = 6) -> None:
    """
    Wrapper function around "train_degs", that is able to handle different parameters for ridge and
    lasso and automatically prints the resulting diagrams.
    :param deg: maximum degree for which to train
    :param solver: ridge or lasso
    :param x: x vector
    :param y: y vector
    :param z: z vector
    :param task: task this belongs to, to sort the figures into the correct folder
    :return: None
    """
    if parameter == "epoch":
        test_r, train_r, test_m, train_m, beta = loop_over_epochs(deg=deg, values=values,
                                                                  solver=solver, x=x, y=y, z=z,
                                                                  la=la, batches=batches, g0=g0)
    elif parameter == "batch":
        test_r, train_r, test_m, train_m, beta = loop_over_batches(deg=deg, values=values,
                                                                   solver=solver, x=x, y=y, z=z,
                                                                   epochs=epochs, la=la, g0=g0)
    elif parameter == "lambda":
        test_r, train_r, test_m, train_m, beta = loop_over_lambda(deg=deg, values=values,
                                                                  solver=solver, x=x, y=y, z=z,
                                                                  epochs=epochs, batches=batches, g0=g0)
    elif parameter == "learning_rate":
        test_r, train_r, test_m, train_m, beta = loop_over_learning_rate(deg=deg, values=values,
                                                                  solver=solver, x=x, y=y, z=z,
                                                                  epochs=epochs, batches=batches, la=la)
    else:
        print("invalid parameter given")
        sys.exit(-2)
    # refactoring shape of error and label vectors
    errors = np.append(test_m, train_m, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    # https://stackoverflow.com/a/952952
    labels = [item for sublist in labels for item in sublist]
    # plotting mean squared error
    if parameter=="lambda" or parameter == "learning_rate":
        logx = True
    else:
        logx = False
    print_errors(values, errors, labels,
                 task + "/" + solver + "_mse_parameter_"+parameter, logx=logx, logy=True,
                 xlabel=parameter, d=d, ylabel="mean squared error")
    errors = np.append(test_r, train_r, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test R^2 " + str(i + 1)])
        labels[1].extend(["train R^2 " + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(values, errors, labels,
                 task + "/" + solver + "_r_squared_parameter_"+parameter, logx=logx, logy=True,
                 xlabel=parameter, d=d, ylabel="R^2 value")


def train_print_single_lambda(deg: int, epochs: int, batches: int, la: float = 0,
                              solver: str = "ridge", x: np.array = None, y: np.array = None,
                              z: np.array = None, task: str = "a", d: int = 6):
    """
    Same as "train_print_diff_lambdas" but only for a single lambda
    :param deg: maximum degree for which to train
    :param batches:
    :param epochs:     :param la: lambda for which to plot
    :param solver: ridge or lasso
    :param x: x vector
    :param y: y vector
    :param z: z vector
    :param task: task this belongs to, to sort the figures into the correct folder
    :param d:
    :return: None
    """
    if task == "f" or task == "e":
        logy = False
    else:
        logy = True
    test_r, train_r, test_m, train_m, beta = train_degs(maxdeg=deg, solver=solver, la=la, x=x, y=y,
                                                        z=z, epochs=epochs, batches=batches)
    errors = [test_m[:, -1], train_m[:, -1]]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 task + "/" + solver + "_mse_deg" + str(deg) + "_epochs_" +
                 str(epochs) + "_batches_" + str(batches) + "_lambda_" + str(la),
                 logy=True, ylabel="mean squared error", d=d)
    errors = [test_r[:, -1], train_r[:, -1]]
    labels = ["test R^2 ", "train R^2 "]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 task + "/" + solver + "_R_squared_deg" + str(deg) + "_epochs_" +
                 str(epochs) + "_batches_" + str(batches) + "_lambda_" + str(la),
                 logy=logy, ylabel="R squared value", d=d)


def task_a():
    deg = 10
    epochs = EPOCHS
    batches = BATCHES
    g0 = LEARNING_RATE
    test_r, train_r, test_m, train_m, beta = train_degs(deg, epochs=epochs, batches=batches, g0=g0)
    print_cont(deg, beta[:, :, -1], "a/ols_betas_epochs_" + str(epochs) + "_batches_" + str(batches),
               "Parameter values")
    errors = [test_m[:, -1], train_m[:, -1]]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_mse_epochs_" + str(epochs) + "_batches_" + str(batches),
                 logy=True, ylabel="mean squared error", d=4)
    errors = [test_r[:, -1], train_r[:, -1]]
    labels = ["test R^2", "train R^2"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_R_squared_epochs_" + str(epochs) + "_batches_" + str(batches),
                 ylabel="R^2 value", d=4)
    deg = 10
    minorder = -6
    maxorder = 0
    order = np.array([minorder, maxorder])
    n = int(np.linalg.norm(order, 1) * 10)
    lrs = np.logspace(order[0], order[1], num=n)
    train_print_hyperparameter(deg=deg, values=lrs, parameter="learning_rate",
                               solver="ols", epochs=epochs, batches=batches, task="a", d=6)
    g0 = 2e-3
    btchs = np.linspace(1, 50, 50, dtype=int)
    train_print_hyperparameter(deg=deg, values=btchs, parameter="batch", g0=g0,
                               solver="ols", epochs=epochs, task="a", d=6)
    batches = 35
    epcs = np.linspace(1, 250, 250, dtype=int)
    train_print_hyperparameter(deg=deg, values=epcs, parameter="epoch", g0=g0,
                               solver="ols", batches=batches, task="a", d=6)
    epochs = 50
    minorder = -4
    maxorder = 4
    order = np.array([minorder, maxorder])
    n = int(np.linalg.norm(order, 1) * 5)
    lambdas = np.logspace(order[0], order[1], num=n)
    train_print_hyperparameter(deg=deg, values=lambdas, parameter="lambda", g0=g0,
                               solver="ridge", epochs=epochs, batches=batches, task="a", d=6)




NOISE_LEVEL = 0.1
MAX_DEG = 30
RESOLUTION = .02
RANDOM_SEED = 1337
SCALE_DATA = True
DATA_FILE = "files/test.csv"
np.random.seed(RANDOM_SEED)
MAX_ITER = 100000

EPOCHS = 100
BATCHES = 20
LEARNING_RATE = 1e-3

task_a()
EPOCHS = 50
BATCHES = 30
LEARNING_RATE = 2e-3
