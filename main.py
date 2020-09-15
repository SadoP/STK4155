import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import colormaps as cmaps


def franke(x, y, noise_level):
    if x.any() < 0 or x.any() > 1 or y.any() < 0 or y.any() > 1:
        print("Franke function is only valid for x and y between 0 and 1.")
        return
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    noise = np.random.normal(0, 1, term4.shape)
    return term1 + term2 + term3 + term4 + noise_level*noise


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
    X = np.array([x, y]).transpose()
    poly = PolynomialFeatures(p)
    return poly.fit_transform(X)


def solve_lin_equ(y, x, solver="least_squares"):
    if solver == "least_squares":
        var = np.linalg.inv(x.transpose() @ x)
        beta = np.linalg.pinv(x) @ y
    elif solver == "lasso":
        var = np.linalg.inv()
        beta = np.linalg.pinv()
    return beta, var.diagonal()


def create_grid(res):
    x = np.arange(0, 1, res)
    y = np.arange(0, 1, res)
    x, y = np.meshgrid(x, y)
    return x, y


def create_data(deg, cross_validation=1):
    x, y = create_grid(RESOLUTION)
    X = coords_to_polynomial(x, y, deg)
    Z = franke(x, y, NOISE_LEVEL).flatten()
    if SCALE_DATA:
        X = scale(X, axis=1)
    n = int(X.shape[0])
    m = int(X.shape[1])
    if cross_validation == 1:
        k = 1
        l = int(np.floor(n*0.2))
        train_l = int(np.floor(n*0.8))
    else:
        k = int(cross_validation)
        l = int(np.floor(n / k))
        train_l = l*(k-1)
    test_l = l
    X, Z = shuffle(X, Z, random_state=RANDOM_SEED)
    X_train = np.zeros(shape=(k, train_l, m))
    X_test = np.zeros(shape=(k, test_l, m))
    Z_train = np.zeros(shape=(k, train_l))
    Z_test = np.zeros(shape=(k, test_l))
    for i in range(k):
        X_train[i, :, :] = np.delete(X, np.arange(i*l, (i+1)*l, 1, int), axis=0)
        X_test[i, :, :] = X[i*l:(i+1)*l, :]
        Z_train[i, :] = np.delete(Z, np.arange(i*l, (i+1)*l, 1, int), axis=0)
        Z_test[i, :] = Z[i*l:(i+1)*l]
    X_train = np.transpose(X_train, (1, 2, 0))
    X_test = np.transpose(X_test, (1, 2, 0))
    Z_train = np.transpose(Z_train, (1, 0))
    Z_test = np.transpose(Z_test, (1, 0))
    return X_train, X_test, Z_train, Z_test


def predict(beta, X):
    return X @ beta


def err_from_var(var, ssize):
    return 1.97 * np.sqrt(var) / np.sqrt(ssize)


def train(deg, cross_validation=1, bootstraps=0):
    X_train, X_test, Z_train, Z_test = create_data(deg, cross_validation)
    testRs = np.zeros(shape=cross_validation)
    testMs = np.zeros(shape=cross_validation)
    trainRs = np.zeros(shape=cross_validation)
    trainMs = np.zeros(shape=cross_validation)
    for i in range(cross_validation):
        beta, var = solve_lin_equ(Z_train[:, i].flatten(), X_train[:, :, i])
        err = err_from_var(var, len(Z_train[:, i]))
        if bootstraps > 0:
            L = X_test.shape[0]
            inds = np.random.randint(0, high=L, size=bootstraps)
            testx = X_test[inds, :, i]
            trainx = X_train[inds, :, i]
            testz = Z_test[inds, i]
            trainz = Z_train[inds, i]
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
    return int((maxdeg+2)*(maxdeg+1)/2)


def train_degs(maxdeg, cross_validation=1, bootstraps=0):
    beta = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    var = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    err = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    test_R = np.zeros(shape=(maxdeg, 1))
    train_R = np.zeros(shape=(maxdeg, 1))
    test_M = np.zeros(shape=(maxdeg, 1))
    train_M = np.zeros(shape=(maxdeg, 1))
    for i in range(maxdeg):
        t = train(i+1, cross_validation=cross_validation, bootstraps=bootstraps)
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
    return test_R, train_R, test_M, train_M, beta, var, err


def print_errors(deg, errors, labels, name, log=False):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    x_values = np.linspace(1, deg, deg)
    for i in range(len(errors)):
        plt.plot(x_values, errors[i], label=labels[i])
    plt.legend()
    ax = fig.gca()
    ax.set_xlabel("Degree")
    ax.set_ylabel("error value")
    plt.xticks(range(1, deg+1))
    if log:
        plt.yscale('log')
    fig.savefig("images/"+name+".png", dpi=300)


def print_cont(deg, data, name):
    x = np.linspace(1, maxMatLen(deg), maxMatLen(deg))
    y = np.linspace(1, deg, deg)
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, data, cmap=cmaps.parula)
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Polynomial Degree")
    ax.set_zlabel(name + " value")
    fig.savefig("images/"+name+".png", dpi=300)


def task_a():
    deg = 5
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    print_cont(deg, err, "error")
    print_cont(deg, beta, "betas")
    print_cont(deg, np.nan_to_num(np.divide(err, beta), nan=0), "errorByBeta")
    errors = [test_M, train_M, test_R, train_R]
    labels = ["test MSE", "train MSE", "test R^2", "train R^2"]
    print_errors(deg, errors, labels, "errors")


def task_b():
    deg = MAX_DEG
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(deg, errors, labels, "errors_highdeg", True)
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg, bootstraps=40)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(deg, errors, labels, "errors_highdeg_bootstrap", True)


def task_c():
    cross_validation = CROSS_VALIDATION
    deg = MAX_DEG
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg, cross_validation=cross_validation)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(deg, errors, labels, "errors_highdeg_crossvalidation", True)


NOISE_LEVEL = 0.2
MAX_DEG = 25
RESOLUTION = .05
RANDOM_SEED = 1337
CROSS_VALIDATION = 5

np.random.seed(RANDOM_SEED)
#plot_franke()
SCALE_DATA = False
#task_a()
SCALE_DATA = True
task_b()
task_c()

X_train, X_test, Z_train, Z_test = create_data(2, 1)
