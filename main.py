import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import colormaps as cmaps
import pandas as pd
from matplotlib import rcParams

def franke(x, y, noise_level):
    if x.any() < 0 or x.any() > 1 or y.any() < 0 or y.any() > 1:
        print("Franke function is only valid for x and y between 0 and 1.")
        return
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    noise = np.random.normal(0, 1, term4.shape)
    return term1 + term2 + term3 + term4 + noise_level * noise


def r2_error(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def mse_error(y_data, y_model):
    return np.sum((y_data - y_model) ** 2) / np.size(y_model)


def coords_to_polynomial(x, y, p):
    if x.shape != y.shape:
        print("mismatch between size of x and y vector")
        return
    x = x.flatten()
    y = y.flatten()
    X = np.zeros(shape=(len(x), maxMatLen(p)))
    k = 0
    for i in range(p + 1):
        for j in range(i + 1):
            X[:, k] = x ** (i - j) * y ** (j)
            k += 1
    return X


def solve_lin_equ(y, x, solver="ols", l=1):
    if solver == "ols":
        var = np.linalg.inv(x.transpose() @ x)
        beta = np.linalg.pinv(x) @ y
    elif solver == "ridge":
        var = np.linalg.inv(x.transpose() @ x + l * np.identity(x.shape[1]))
        beta = var @ x.transpose() @ y
    elif solver == "lasso":
        clf = linear_model.Lasso(alpha=l)
        clf.fit(x, y)
        beta = clf.coef_
        var = np.zeros(shape=(1, 1))
    return beta, np.abs(var.diagonal())


def create_grid(res):
    x = np.arange(0, 1, res)
    y = np.arange(0, 1, res)
    x, y = np.meshgrid(x, y)
    return x, y


def create_data(deg, cross_validation=1, x=None, y=None, Z=None):
    if x is None or y is None:
        x, y = create_grid(RESOLUTION)
        Z = franke(x, y, NOISE_LEVEL).flatten()
    X = coords_to_polynomial(x, y, deg)
    if SCALE_DATA:
        X = scale(X, axis=1)
    if Z is None:
        Z = franke(x, y, NOISE_LEVEL).flatten()
    n = int(X.shape[0])
    m = int(X.shape[1])
    if cross_validation == 1:
        k = 1
        l = int(np.floor(n * 0.2))
        train_l = n - l
    else:
        k = int(cross_validation)
        l = int(np.floor(n / k))
        train_l = l * (k - 1)
    test_l = l
    X, Z = shuffle(X, Z, random_state=RANDOM_SEED)
    X_train = np.zeros(shape=(k, train_l, m))
    X_test = np.zeros(shape=(k, test_l, m))
    Z_train = np.zeros(shape=(k, train_l))
    Z_test = np.zeros(shape=(k, test_l))
    for i in range(k):
        X_train[i, :, :] = np.delete(X, np.arange(i * l, (i + 1) * l, 1, int), axis=0)
        X_test[i, :, :] = X[i * l:(i + 1) * l, :]
        Z_train[i, :] = np.delete(Z, np.arange(i * l, (i + 1) * l, 1, int), axis=0)
        Z_test[i, :] = Z[i * l:(i + 1) * l]
    X_train = np.transpose(X_train, (1, 2, 0))
    X_test = np.transpose(X_test, (1, 2, 0))
    Z_train = np.transpose(Z_train, (1, 0))
    Z_test = np.transpose(Z_test, (1, 0))
    return X_train, X_test, Z_train, Z_test


def predict(beta, X):
    return X @ beta


def err_from_var(var, ssize):
    return 1.97 * np.sqrt(var) / np.sqrt(ssize)


def train(deg, cross_validation=1, bootstraps=0, solver="ols", l=1, x=None, y=None, Z=None):
    X_train, X_test, Z_train, Z_test = create_data(deg, cross_validation, x=x, y=y, Z=Z)
    testRs = np.zeros(shape=cross_validation)
    testMs = np.zeros(shape=cross_validation)
    trainRs = np.zeros(shape=cross_validation)
    trainMs = np.zeros(shape=cross_validation)
    for i in range(cross_validation):
        beta, var = solve_lin_equ(Z_train[:, i].flatten(), X_train[:, :, i], solver=solver, l=l)
        err = err_from_var(var, len(Z_train[:, i]))
        if bootstraps > 0:
            trainL = X_train.shape[0]
            trainInds = np.random.randint(0, high=trainL, size=bootstraps)
            trainx = X_train[trainInds, :, i]
            trainz = Z_train[trainInds, i]
            testx = X_test[:, :, i]
            testz = Z_test[:, i]
        else:
            testx = X_test[:, :, i]
            trainx = X_train[:, :, i]
            testz = Z_test[:, i]
            trainz = Z_train[:, i]
        testRs[i] = r2_error(testz, predict(beta, testx))
        testMs[i] = mse_error(testz, predict(beta, testx))
        trainRs[i] = r2_error(trainz, predict(beta, trainx))
        trainMs[i] = mse_error(trainz, predict(beta, trainx))
    test_R = np.average(testRs)
    train_R = np.average(trainRs)
    test_M = np.average(testMs)
    train_M = np.average(trainMs)
    return beta, var, err, test_R, train_R, test_M, train_M


def maxMatLen(maxdeg):
    return int((maxdeg + 2) * (maxdeg + 1) / 2)


def train_degs(maxdeg, cross_validation=1, bootstraps=0, solver="ols", l=1, x=None, y=None, Z=None):
    beta = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    var = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    err = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    test_R = np.zeros(shape=(maxdeg, 1))
    train_R = np.zeros(shape=(maxdeg, 1))
    test_M = np.zeros(shape=(maxdeg, 1))
    train_M = np.zeros(shape=(maxdeg, 1))
    for i in range(maxdeg):
        print("training degree: " + str(i + 1))
        t = train(i + 1, cross_validation=cross_validation, bootstraps=bootstraps, solver=solver,
                  l=l, x=x, y=y, Z=Z)
        b = t[0]
        beta[i, 0:len(b)] = b
        v = t[1]
        var[i, 0:len(v)] = v
        e = t[2]
        err[i, 0:len(e)] = e
        test_R[i] = t[3]
        train_R[i] = t[4]
        test_M[i] = t[5]
        train_M[i] = t[6]
    return test_R.ravel(), train_R.ravel(), test_M.ravel(), train_M.ravel(), beta, var, err


def print_errors(x_values, errors, labels, name, logy=False, logx=False, xlabel="Degree",
                 ylabel="error value", d=4):
    fig = plt.figure(figsize=(d, d), dpi=300)
    x_values = x_values
    for i in range(len(errors)):
        label = labels[i]
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
    return plt


def print_cont(deg, data, name, zlabel):
    x = np.linspace(1, maxMatLen(deg), maxMatLen(deg))
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


def read_data_file():
    data_file = DATA_FILE
    return pd.read_csv(data_file, delimiter=",")


def get_data():
    """
    I used some data that I had left over from my previous position. This is calculated magnetic
    field data from some permanent magnet configuration. The data does not follow a strictly
    integer-polynomial trend but can be described not too badly by a polynom within the boundaries.
    I made the data more sparse by randomly removing 80% of the original data.
    :return:
    """
    data = read_data_file()
    x = np.array(data["x"])
    y = np.array(data["y"])
    Z = np.array(data["B"])
    cutoff = int(np.floor(len(Z)*.2))
    x, y, Z = shuffle(x, y, Z, random_state=RANDOM_SEED)
    x = x[:cutoff]
    y = y[:cutoff]
    Z = Z[:cutoff]
    Z = Z / np.max(Z)
    return x, y, Z


def print_data():
    x, y, Z = get_data()
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, Z, c='y', marker='o')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("B")
    fig.savefig("images/data.png", dpi=300)


def task_a():
    deg = 5
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    print_cont(deg, err, "a/ols_error_scaling_"+SCALE_DATA.__str__(), "Parameter error")
    print_cont(deg, beta, "a/ols_betas_scaling_"+SCALE_DATA.__str__(), "Parameter values")
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_mse_scaling_"+SCALE_DATA.__str__(),
                 logy=True, ylabel="mean squared error")
    errors = [test_R, train_R]
    labels = ["test R^2", "train R^2"]
    print_errors(np.linspace(1, deg, deg), errors, labels,
                 "a/ols_R_squared_scaling_"+SCALE_DATA.__str__(),
                 ylabel="R^2 value")


def task_b():
    deg = MAX_DEG
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "b/ols_mse_highdeg_noBoot", True,
                 ylabel="mean squared error")
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg, bootstraps=BOOTSTRAPS)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "b/ols_mse_highdeg_Boot_"+str(
        BOOTSTRAPS), True, ylabel="mean squared error")


def task_c():
    deg = MAX_DEG
    cross_validation = 5
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg,
                                                                  cross_validation=cross_validation)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "c/mse_highdeg_crossvalidation_" +
                 str(cross_validation), True, ylabel="mean squared error")
    errors = [test_R, train_R]
    labels = ["test R^2", "train R^2"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "c/r_squared_highdeg_crossvalidation_" +
                 str(cross_validation), True, ylabel="R^2 value")
    cross_validation = 10
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg,
                                                                  cross_validation=cross_validation)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "c/mse_highdeg_crossvalidation_"+str(
        cross_validation), True, ylabel="mean squared error")
    errors = [test_R, train_R]
    labels = ["test R^2", "train R^2"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "c/r_squared_highdeg_crossvalidation_" +
                 str(cross_validation), True, ylabel="R^2 value")


def task_d():
    deg = 10
    cross_validation = CROSS_VALIDATION
    order = np.array([-7, 5])
    N = int(np.linalg.norm(order, 1) * 5)
    test_R = np.zeros(shape=(deg, N))
    train_R = np.zeros(shape=(deg, N))
    test_M = np.zeros(shape=(deg, N))
    train_M = np.zeros(shape=(deg, N))
    lambdas = np.logspace(order[0], order[1], num=N)
    i = 0
    for l in lambdas:
        test_R[:, i], train_R[:, i], test_M[:, i], train_M[:, i], beta, var, err = train_degs(
            maxdeg=deg, cross_validation=cross_validation, solver="ridge", l=l)
        i = i + 1
    errors = np.append(test_M, train_M, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "d/ridge_mse_cross_"+str(cross_validation), True, True,
                 xlabel="lambda", d=7, ylabel="mean squared error")
    errors = np.append(test_R, train_R, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test R^2 " + str(i + 1)])
        labels[1].extend(["train R^2 " + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "d/ridge_R_squared_cross_"+str(cross_validation), True,
                 True, xlabel="lambda", d=7, ylabel="R^2 value")

    bootstraps = BOOTSTRAPS
    test_R = np.zeros(shape=(deg, N))
    train_R = np.zeros(shape=(deg, N))
    test_M = np.zeros(shape=(deg, N))
    train_M = np.zeros(shape=(deg, N))
    lambdas = np.logspace(order[0], order[1], num=N)
    i = 0
    for l in lambdas:
        test_R[:, i], train_R[:, i], test_M[:, i], train_M[:, i], beta, var, err = train_degs(
            maxdeg=deg, bootstraps=bootstraps, solver="ridge", l=l)
        i = i + 1
    errors = np.append(test_M, train_M, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "d/ridge_mse_boot_"+str(bootstraps), True, True,
                 xlabel="lambda", d=7, ylabel="mean squared error")
    errors = np.append(test_R, train_R, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test R^2 " + str(i + 1)])
        labels[1].extend(["train R^2 " + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "d/ridge_R_squared_boot_"+str(bootstraps), True,
                 True, xlabel="lambda", d=7, ylabel="R^2 value")

    best_l = 1
    deg = MAX_DEG*2
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg, bootstraps=bootstraps,
                                                                  solver="ridge", l=best_l)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "d/ridge_mse_highdeg_Boot_"+str(
        bootstraps)+"_lambda_"+str(best_l), True, ylabel="mean squared error")
    errors = [test_R, train_R]
    labels = ["test R^2 ", "train R^2 "]
    print_errors(np.linspace(1, deg, deg), errors, labels, "d/ridge_R_squared_highdeg_Boot_"+str(
        bootstraps)+"_lambda_"+str(best_l), True, ylabel="mean squared error")



def task_e():
    deg = 5
    order = np.array([-3, 0])
    N = int(np.linalg.norm(order, 1) * 50)
    lambdas = np.logspace(order[0], order[1], num=N)
    test_R = np.zeros(shape=(deg, N))
    train_R = np.zeros(shape=(deg, N))
    test_M = np.zeros(shape=(deg, N))
    train_M = np.zeros(shape=(deg, N))
    i = 0
    for l in lambdas:
        test_R[:, i], train_R[:, i], test_M[:, i], train_M[:, i], beta, var, err = train_degs(
            maxdeg=deg, cross_validation=CROSS_VALIDATION, solver="lasso", l=l)
        i = i + 1
    errors = np.append(test_M, train_M, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    # https://stackoverflow.com/a/952952
    print_errors(lambdas, errors, labels, "errors_lasso_crossvalidation", True, True,
                 xlabel="lambda")
    deg = MAX_DEG
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(maxdeg=deg, bootstraps=400,
                                                                  solver="lasso", l=BEST_L)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(np.linspace(1, deg, deg), errors, labels, "errors_highdeg_lasso", True)


def task_f():
    #print_data()
    x, y, Z = get_data()

    deg = 35
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(maxdeg=deg, cross_validation=CROSS_VALIDATION,
                                                                  bootstraps=0,
                                                                  solver="ols", l=1, x=x, y=y,
                                                                  Z=Z)
    print_errors(np.linspace(1, deg, deg), np.array([test_M, train_M]), ["test MSE",
                                                                         "train MSE"],
                 "mse_custom_data_ols", logy=True)
    print_errors(np.linspace(1, deg, deg), np.array([test_R, train_R]), ["test R^2",
                                                                         "train R^2"],
                 "r_squared_custom_data_ols", logy=True, ylabel="R^2")


    deg = 10
    order = np.array([-4, 8])
    N = int(np.linalg.norm(order, 1) * 5)
    test_R = np.zeros(shape=(deg, N))
    train_R = np.zeros(shape=(deg, N))
    test_M = np.zeros(shape=(deg, N))
    train_M = np.zeros(shape=(deg, N))
    lambdas = np.logspace(order[0], order[1], num=N)
    i = 0
    for l in lambdas:
        print(l)
        test_R[:, i], train_R[:, i], test_M[:, i], train_M[:, i], beta, var, err = train_degs(
            maxdeg=deg, cross_validation=CROSS_VALIDATION, bootstraps=0, solver="ridge", l=l, x=x, y=y, Z=Z)
        i = i + 1
    errors = np.append(test_M, train_M, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "mse_custom_data_ridge", True, True,
                 xlabel="lambda")
    errors = np.append(test_R, train_R, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test R^2 " + str(i + 1)])
        labels[1].extend(["train R^2 " + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "r_squared_custom_data_ridge", True, True,
                 xlabel="lambda", ylabel="R^2")

    deg = 10
    order = np.array([-7, 3])
    N = int(np.linalg.norm(order, 1) * 5)
    test_R = np.zeros(shape=(deg, N))
    train_R = np.zeros(shape=(deg, N))
    test_M = np.zeros(shape=(deg, N))
    train_M = np.zeros(shape=(deg, N))
    lambdas = np.logspace(order[0], order[1], num=N)
    i = 0
    for l in lambdas:
        print(l)
        test_R[:, i], train_R[:, i], test_M[:, i], train_M[:, i], beta, var, err = train_degs(
            maxdeg=deg, cross_validation=CROSS_VALIDATION, bootstraps=0, solver="lasso", l=l, x=x, y=y, Z=Z)
        i = i + 1
    errors = np.append(test_M, train_M, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test MSE" + str(i + 1)])
        labels[1].extend(["train MSE" + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "mse_custom_data_lasso", True, True,
                 xlabel="lambda")
    errors = np.append(test_R, train_R, axis=0)
    labels = [[], []]
    for i in range(deg):
        labels[0].extend(["test R^2 " + str(i + 1)])
        labels[1].extend(["train R^2 " + str(i + 1)])
    labels = [item for sublist in labels for item in sublist]
    print_errors(lambdas, errors, labels, "r_squared_custom_data_lasso", True, True,
                 xlabel="lambda", ylabel="R^2")


def show_cond_numbers():
    x, y = create_grid(0.1)
    X3 = coords_to_polynomial(x, y, 3)
    X5 = coords_to_polynomial(x, y, 5)
    X10 = coords_to_polynomial(x, y, 10)
    c3 = np.linalg.cond(X3.transpose()@X3)
    c5 = np.linalg.cond(X5.transpose()@X5)
    c10 = np.linalg.cond(X10.transpose()@X10)
    X3 = scale(X3, axis=1)
    X5 = scale(X5, axis=1)
    X10 = scale(X10, axis=1)
    c3s = np.linalg.cond(X3.transpose()@X3)
    c5s = np.linalg.cond(X5.transpose()@X5)
    c10s = np.linalg.cond(X10.transpose()@X10)
    print("condition numbers for different degress of the polynomial:\n"
          "deg3: %f\ndeg5: %f\ndeg10:%f\nSame for centered data:\ndeg3: %f\ndeg5: %f\ndeg10:%f" % (
          c3, c5, c10, c3s, c5s, c10s))

NOISE_LEVEL = 0.1
MAX_DEG = 30
RESOLUTION = .02
RANDOM_SEED = 1337
BOOTSTRAPS = 200
BEST_L = 1e-2
DATA_FILE = "files/test.csv"
np.random.seed(RANDOM_SEED)
CROSS_VALIDATION = 5

show_cond_numbers()
SCALE_DATA = False
"""
task_a()
SCALE_DATA = True
task_a()
task_b()
"""
SCALE_DATA = True
#task_c()
RESOLUTION = .1
task_d()
# task_e()
# task_f()
