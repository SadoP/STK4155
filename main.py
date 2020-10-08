from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn import linear_model
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


def plot_franke():
    """
    plots the franke function for x,y between 0 and 1
    """
    x, y = create_grid(RESOLUTION)
    z = franke(x, y, 0)
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cmaps.parula, linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig("images/franke.png")


def r2_error(y_data: np.array, y_model: np.array) -> float:
    """
    Calculates R squared based on predicted and true data.
    """
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def mse_error(y_data: np.array, y_model: np.array) -> float:
    """
    Calculates mean squared error based in predicted and true data
    """
    return np.sum((y_data - y_model) ** 2) / np.size(y_model)


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


def solve_lin_equ(y: np.array, x: np.array, solver: str = "ols", la: float = 1) -> Tuple[
        np.array, np.array]:
    """
    Solves linear equation of type y = x*beta. This can be done using ordinary least squares (OLS),
    ridge or lasso regression. For ridge and lasso, an additional parameter l for shrinkage /
    normlization needs to be provided.
    :param y: solution vector of linear problem
    :param x: feature vector or matrix of problem
    :param solver: solver. Can be ols, ridge or lasso
    :param la: shrinkage / normalization parameter for ridge or lasso
    :return: parameter vector that solves the problem & variance of parameter vector
    """
    beta = np.zeros(x.shape[0])
    var = np.zeros(beta.shape)
    if solver == "ols":
        var = np.linalg.inv(x.transpose() @ x)
        beta = np.linalg.pinv(x) @ y
    elif solver == "ridge":
        var = np.linalg.inv(x.transpose() @ x + la * np.identity(x.shape[1]))
        beta = var @ x.transpose() @ y
    elif solver == "lasso":
        clf = linear_model.Lasso(alpha=la, max_iter=MAX_ITER)
        clf.fit(x, y)
        beta = clf.coef_
        var = np.zeros(shape=(1, 1))
    return beta, np.abs(var.diagonal())


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


def create_data(deg: int, cross_validation: int = 1, x: np.array = None, y: np.array = None,
                z: np.array = None) \
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
    :param cross_validation: Number of folds for cross validation. Default: 1. This means a split
        in 80:20 training:test without cross validation
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
    m = int(feature_mat.shape[1])
    # k will be the amount of folds
    # test_l is the length of the test split
    # train_l is the length of the training split
    if cross_validation == 1:
        # 80:20 split in train and test data
        k = 1
        test_l = int(np.floor(n * 0.2))
        train_l = n - test_l
    else:
        # split into k folds, length depends on amount of folds
        k = cross_validation
        test_l = int(np.floor(n / k))
        train_l = n - test_l
    # randomizing order of feature matrix and result vector and allocating split variables
    feature_mat, z = shuffle(feature_mat, z, random_state=RANDOM_SEED)
    feature_mat_train = np.zeros(shape=(k, train_l, m))
    feature_mat_test = np.zeros(shape=(k, test_l, m))
    z_train = np.zeros(shape=(k, train_l))
    z_test = np.zeros(shape=(k, test_l))
    for i in range(k):
        # For each split the first test_l amount of data is removed from the training set. The same
        # array of data is then the test set for this fold.
        feature_mat_train[i, :, :] = np.delete(feature_mat,
                                               np.arange(i * test_l, (i + 1) * test_l, 1, int),
                                               axis=0)
        feature_mat_test[i, :, :] = feature_mat[i * test_l:(i + 1) * test_l, :]
        z_train[i, :] = np.delete(z, np.arange(i * test_l, (i + 1) * test_l, 1, int), axis=0)
        z_test[i, :] = z[i * test_l:(i + 1) * test_l]
    # reshaping to make it easier in the coming functions. The first two indeces are left in their
    # previous relative and are the length and height of the feautres. The now last index represents
    # the fold.
    feature_mat_train = np.transpose(feature_mat_train, (1, 2, 0))
    feature_mat_test = np.transpose(feature_mat_test, (1, 2, 0))
    z_train = np.transpose(z_train, (1, 0))
    z_test = np.transpose(z_test, (1, 0))
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


def train(deg: int, cross_validation: int = 1, bootstraps: int = 0, solver: str = "ols",
          la: float = 1,
          x: np.array = None, y: np.array = None, z: np.array = None) -> \
        Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Calculates fit based on given degree. If cross validation of bootstraps are given, the
    corresponding algorithms will be applied. If noe data for x,y and z are given, data will be
    created from / for the franke function, otherwise the provided data will be used
    :param deg: Degree of the polynomial to be fitted onto
    :param cross_validation: amount of folds for cross validation. Default:1 means 80:20 split
    :param bootstraps: amount of bootstraps for bootstrap evaluation
    :param solver: Algorithm for solving. Can be ols, ridge or lasso
    :param la: parameter for ridge and lasso regression
    :param x: x-vector
    :param y: y-vector
    :param z: z-vector
    :return: Regression parameter vector, error and variance of parameter vector, test & training
        data R squqared value and mean squared error
    """
    # creates feature matrix and result vector
    feature_mat_train, feature_mat_test, z_train, z_test = create_data(deg, cross_validation, x=x,
                                                                       y=y, z=z)
    # declares variables for results
    test_rs = np.zeros(shape=cross_validation)
    test_ms = np.zeros(shape=cross_validation)
    train_rs = np.zeros(shape=cross_validation)
    train_ms = np.zeros(shape=cross_validation)
    beta = np.zeros(feature_mat_test.shape[1])
    var = np.zeros(beta.shape)
    err = np.zeros(beta.shape)
    # looping over folds for cross validation
    for i in range(cross_validation):
        # solving the linear equation, calculating error for parameter vector
        beta, var = solve_lin_equ(z_train[:, i].flatten(), feature_mat_train[:, :, i],
                                  solver=solver, la=la)
        err = err_from_var(var, len(z_train[:, i]))
        if bootstraps > 0:
            # If bootstrap resampling is to be evaluated, randomly select samples from the training
            # data. The index variable may contain the same values several times
            train_l = feature_mat_train.shape[0]
            train_inds = np.random.randint(0, high=train_l, size=bootstraps)
            train_x = feature_mat_train[train_inds, :, i]
            train_z = z_train[train_inds, i]
            test_x = feature_mat_test[:, :, i]
            test_z = z_test[:, i]
        else:
            # No bootstrap: Every result is part of the sample
            test_x = feature_mat_test[:, :, i]
            train_x = feature_mat_train[:, :, i]
            test_z = z_test[:, i]
            train_z = z_train[:, i]
        # Calculate errors based on the chosen samples
        test_rs[i] = r2_error(test_z, predict(beta, test_x))
        test_ms[i] = mse_error(test_z, predict(beta, test_x))
        train_rs[i] = r2_error(train_z, predict(beta, train_x))
        train_ms[i] = mse_error(train_z, predict(beta, train_x))
    # average over the folds
    test_r = np.average(test_rs)
    train_r = np.average(train_rs)
    test_m = np.average(test_ms)
    train_m = np.average(train_ms)
    return test_r, train_r, test_m, train_m, beta, var, err


def max_mat_len(maxdeg: int) -> int:
    """
    Calculates the size of the polyonmial based on its degree
    :param maxdeg: degree of polynomial
    :return: size of the polynomial
    """
    return int((maxdeg + 2) * (maxdeg + 1) / 2)


def train_degs(maxdeg: int, cross_validation: int = 1, bootstraps: int = 0, solver: str = "ols",
               la: float = 1,
               x: np.array = None, y: np.array = None, z: np.array = None) \
        -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    This is a wrapper for the function "train", that loops over all the degrees given in maxdeg
    :param maxdeg: highest order degree to be trained
    :param cross_validation: amount of folds for cross validation. Default:1 means 80:20 split
    :param bootstraps: amount of bootstraps for bootstrap evaluation
    :param solver: Algorithm for solving. Can be ols, ridge or lasso
    :param la: parameter for ridge and lasso regression
    :param x: x-vector
    :param y: y-vector
    :param z: z-vector
    :return: Regression parameter vector, error and variance of parameter vector, test & training
        data R squqared value and mean squared error
    """
    # declare variables so that the values can be assigned in the loop
    beta = np.zeros(shape=(maxdeg, max_mat_len(maxdeg)))
    var = np.zeros(shape=(maxdeg, max_mat_len(maxdeg)))
    err = np.zeros(shape=(maxdeg, max_mat_len(maxdeg)))
    test_r = np.zeros(shape=(maxdeg, 1))
    train_r = np.zeros(shape=(maxdeg, 1))
    test_m = np.zeros(shape=(maxdeg, 1))
    train_m = np.zeros(shape=(maxdeg, 1))
    # looping over every degree to be trained
    for i in range(maxdeg):
        print("training degree: " + str(i + 1))
        t = train(i + 1, cross_validation=cross_validation, bootstraps=bootstraps, solver=solver,
                  la=la, x=x, y=y, z=z)
        b = t[4]
        beta[i, 0:len(b)] = b
        v = t[5]
        var[i, 0:len(v)] = v
        e = t[6]
        err[i, 0:len(e)] = e
        test_r[i] = t[0]
        train_r[i] = t[1]
        test_m[i] = t[2]
        train_m[i] = t[3]
    return test_r.ravel(), train_r.ravel(), test_m.ravel(), train_m.ravel(), beta, var, err


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
    fig.tight_layout()
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


def train_print_diff_lambdas(deg: int, minorder: int, maxorder: int, cross_validation: int = 1,
                             bootstraps: int = 0,
                             solver: str = "ridge", x: np.array = None, y: np.array = None,
                             z: np.array = None,
                             task: str = "a", d: int = 6) -> None:
    """
    Wrapper functino around "train_degs", that is able to handle different parameters for ridge and
    lasso and automatically prints the resulting diagramms.
    :param deg: maximum degree for which to train
    :param minorder: minimum order of magnitude for parameter
    :param maxorder: maximum order of magnitude for parameter
    :param cross_validation: folds for cross validation
    :param bootstraps: amount of bootstraps
    :param solver: ridge or lasso
    :param x: x vector
    :param y: y vector
    :param z: z vector
    :param task: task this belongs to, to sort the figures into the correct folder
    :param d: size of the figure
    :return: None
    """
    if task == "f" or task == "e":
        logy = False
    else:
        logy = True
    order = np.array([minorder, maxorder])
    n = int(np.linalg.norm(order, 1) * 5)
    test_r = np.zeros(shape=(deg, n))
    train_r = np.zeros(shape=(deg, n))
    test_m = np.zeros(shape=(deg, n))
    train_m = np.zeros(shape=(deg, n))
    # different parameters created based on the orders given before, equally spaced on a logarithmic
    # scale
    lambdas = np.logspace(order[0], order[1], num=n)
    i = 0
    # iterating over lambdas
    for la in lambdas:
        test_r[:, i], train_r[:, i], test_m[:, i], train_m[:, i], beta, var, err = train_degs(
            maxdeg=deg,
            cross_validation=cross_validation,
            bootstraps=bootstraps,
            solver=solver, la=la, x=x,
            y=y, z=z)
        i = i + 1
    # refactoring shape of error and label vectors
    errors = np.append(test_m, train_m, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test deg:" + str(i + 1)])
        labels[1].extend(["train deg:" + str(i + 1)])
    # https://stackoverflow.com/a/952952
    labels = [item for sublist in labels for item in sublist]
    # plotting mean squared error
    print_errors(lambdas, errors, labels,
                 task + "/" + solver + "_mse_cross_" + str(cross_validation) + "_boot_" + str(
                     bootstraps), logx=True, logy=True,
                 xlabel="lambda", d=d, ylabel="mean squared error")
    errors = np.append(test_r, train_r, axis=0)
    print_errors(lambdas, errors, labels,
                 task + "/" + solver + "_R_squared_cross_" + str(cross_validation) + "_boot_" + str(
                     bootstraps), logx=True, logy=logy,
                 xlabel="lambda", d=d, ylabel="R^2 value")


def train_print_single_lambda(deg: int, la: float = 0, cross_validation: int = 1,
                              bootstraps: int = 0,
                              solver: str = "ridge", x: np.array = None, y: np.array = None,
                              z: np.array = None,
                              task: str = "a", d: int = 6):
    """
    Same as "train_print_diff_lambdas" but only for a single lambda
    :param deg: maximum degree for which to train
    :param la: lambda for which to plot
    :param cross_validation: folds for cross validation
    :param bootstraps: amount of bootstraps
    :param solver: ridge or lasso
    :param x: x vector
    :param y: y vector
    :param z: z vector
    :param task: task this belongs to, to sort the figures into the correct folder
    :param d: size of the figure
    :return: None
    """
    if task == "f" or task == "e":
        logy = False
    else:
        logy = True
    test_r, train_r, test_m, train_m, beta, var, err = train_degs(maxdeg=deg,
                                                                  cross_validation=cross_validation,
                                                                  bootstraps=bootstraps,
                                                                  solver=solver, la=la,
                                                                  x=x, y=y, z=z)
    errors = [test_m, train_m]
    labels = ["test", "train"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 task + "/" + solver + "_mse_deg" + str(deg) + "_cross_" +
                 str(cross_validation) + "_Boot_" + str(bootstraps) + "_lambda_" + str(la),
                 logy=True, ylabel="mean squared error", d=d)
    errors = [test_r, train_r]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 task + "/" + solver + "_R_squared_deg" + str(deg) + "_cross_" +
                 str(cross_validation) + "_Boot_" + str(bootstraps) + "_lambda_" + str(la),
                 logy=logy, ylabel="R^2 value", d=d)


def task_a():
    deg = 5
    test_r, train_r, test_m, train_m, beta, var, err = train_degs(deg)
    print_cont(deg, err, "a/ols_error_scaling_" + SCALE_DATA.__str__(), "Parameter error")
    print_cont(deg, beta, "a/ols_betas_scaling_" + SCALE_DATA.__str__(), "Parameter values")
    errors = [test_m, train_m]
    labels = ["test", "train"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_mse_scaling_" + SCALE_DATA.__str__(),
                 logy=True, ylabel="mean squared error", d=4)
    errors = [test_r, train_r]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_R_squared_scaling_" + SCALE_DATA.__str__(),
                 ylabel="R^2 value", d=4)


def task_b():
    deg = MAX_DEG
    bootstraps = np.array([0, BOOTSTRAPS])
    for bs in bootstraps:
        train_print_single_lambda(deg=deg, bootstraps=bs, solver="ols", task="b", d=4)


def task_c():
    deg = MAX_DEG
    cross_validation = np.array([5, 10])
    for cr in cross_validation:
        train_print_single_lambda(deg=deg, cross_validation=cr, task="c", solver="ols", d=4)


def task_d():
    deg = 10
    bootstraps = BOOTSTRAPS
    cross_validation = CROSS_VALIDATION
    minorder = -7
    maxorder = 5
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder,
                             cross_validation=cross_validation,
                             solver="ridge", task="d", d=6)
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder, bootstraps=bootstraps,
                             solver="ridge", task="d", d=6)
    best_l = 1
    train_print_single_lambda(deg=MAX_DEG * 2, la=best_l, bootstraps=bootstraps, solver="ridge",
                              task="d", d=4)


def task_e():
    deg = 10
    cross_validation = CROSS_VALIDATION
    bootstraps = BOOTSTRAPS
    minorder = -4
    maxorder = 0
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder,
                             cross_validation=cross_validation,
                             solver="lasso", task="e", d=6)
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder, bootstraps=bootstraps,
                             solver="lasso", task="e", d=6)
    best_l = 2e-2
    train_print_single_lambda(deg=MAX_DEG * 2, la=best_l, bootstraps=bootstraps, solver="lasso",
                              task="e", d=4)
    best_l = 2e-1
    train_print_single_lambda(deg=MAX_DEG * 2, la=best_l, bootstraps=bootstraps, solver="lasso",
                              task="e", d=4)


def task_f():
    x, y, z = get_data()
    deg = MAX_DEG
    cross_validation = CROSS_VALIDATION
    test_r, train_r, test_m, train_m, _, _, _ = train_degs(deg)
    errors = [test_m, train_m]
    labels = ["test", "train"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "f/ols_mse_scaling_",
                 logy=True, ylabel="mean squared error", d=4)
    errors = [test_r, train_r]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "f/ols_R_squared_scaling_",
                 ylabel="R^2 value", logy=True, d=4)

    deg = 10
    bootstraps = BOOTSTRAPS
    minorder = -1
    maxorder = 7
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder,
                             cross_validation=cross_validation,
                             solver="ridge", task="f", x=x, y=y, z=z, d=6)
    minorder = -6
    maxorder = 0
    train_print_diff_lambdas(deg=deg, minorder=minorder, maxorder=maxorder,
                             cross_validation=cross_validation,
                             solver="lasso", task="f", x=x, y=y, z=z, d=6)
    best_l = 10
    deg = MAX_DEG
    train_print_single_lambda(deg=deg, la=best_l, bootstraps=bootstraps, solver="ridge", task="f",
                              x=x, y=y, z=z, d=4)


def show_cond_numbers():
    """
    Calculates and prints condition numbers for different feature matrices
    :return: None
    """
    x, y = create_grid(0.1)
    x3 = coords_to_polynomial(x, y, 3)
    x5 = coords_to_polynomial(x, y, 5)
    x10 = coords_to_polynomial(x, y, 10)
    c3 = np.linalg.cond(x3.transpose() @ x3)
    c5 = np.linalg.cond(x5.transpose() @ x5)
    c10 = np.linalg.cond(x10.transpose() @ x10)
    x3 = scale(x3, axis=1)
    x5 = scale(x5, axis=1)
    x10 = scale(x10, axis=1)
    c3s = np.linalg.cond(x3.transpose() @ x3)
    c5s = np.linalg.cond(x5.transpose() @ x5)
    c10s = np.linalg.cond(x10.transpose() @ x10)
    print("condition numbers for different degress of the polynomial:\n"
          "deg3: %f\ndeg5: %f\ndeg10:%f\nSame for centered data:\ndeg3: %f\ndeg5: %f\ndeg10:%f" % (
              c3, c5, c10, c3s, c5s, c10s))


NOISE_LEVEL = 0.1
MAX_DEG = 30
RESOLUTION = .02
RANDOM_SEED = 1337
BOOTSTRAPS = 200
BEST_L = 1e-2
DATA_FILE = "files/data.csv"
np.random.seed(RANDOM_SEED)
CROSS_VALIDATION = 5
MAX_ITER = 100000

show_cond_numbers()
SCALE_DATA = False
plot_franke()

task_a()
SCALE_DATA = True
task_a()
task_b()
task_c()
RESOLUTION = .05
task_d()
task_e()
MAX_ITER = 1000
task_f()
