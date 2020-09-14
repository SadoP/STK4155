import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
import colormaps as cmaps


def plot_franke(own_x=np.array([]), own_y=np.array([]), own_z=np.array([])):
    x, y = create_grid(.01)
    z = franke(x, y, 0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    if own_z.size > 0:
        surf = ax.plot_surface(own_x, own_y, (franke(own_x, own_y, 0)-own_z.reshape(own_y.shape)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    else:
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


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


def solve_lin_equ(y, x):
    var = np.linalg.inv(x.transpose() @ x)
    beta = np.linalg.pinv(x) @ y
    return beta, var.diagonal()


def create_grid(res):
    x = np.arange(0, 1, res)
    y = np.arange(0, 1, res)
    x, y = np.meshgrid(x, y)
    return x, y


def create_data(deg):
    x, y = create_grid(RESOLUTION)
    """
    I know this is not the most elegant way and the proper way would be to first calculated a total Matrix X and then
    split it into test and training data, but I want to be able to plot the results of the test and training data, 
    for which I will need to have the x- and y- coordinates before I did the split.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    X_train = coords_to_polynomial(x_train, y_train, deg)
    Z_train = franke(x_train, y_train, NOISE_LEVEL).flatten()
    X_test = coords_to_polynomial(x_test, y_test, deg)
    Z_test = franke(x_test, y_test, NOISE_LEVEL).flatten()
    if(SCALE_DATA):
        X_train = scale(X_train, axis=1)
        X_test  = scale(X_test, axis=1)
    return X_train, X_test, Z_train, Z_test, x_train, x_test, y_train, y_test


def predict(beta, X):
    return X @ beta


def err_from_var(var, ssize):
    return 1.97 * np.sqrt(var) / np.sqrt(ssize)


def train(deg):
    X_train, X_test, Z_train, Z_test, x_tr, x_te, y_tr, y_te = create_data(deg)
    beta, var = solve_lin_equ(Z_train.flatten(), X_train)
    err = err_from_var(var, len(Z_train))
    test_R = r2_error(Z_test, predict(beta, X_test))
    test_M = mse_error(Z_test, predict(beta, X_test))
    train_R = r2_error(Z_train, predict(beta, X_train))
    train_M = mse_error(Z_train, predict(beta, X_train))
    #plot_franke(x_tr, y_tr, predict(beta, X_train))
    return beta, var, err, test_R, train_R, test_M, train_M


def maxMatLen(maxdeg):
    return int((maxdeg+2)*(maxdeg+1)/2)


def train_degs(maxdeg):
    beta = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    var = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    err = np.zeros(shape=(maxdeg, maxMatLen(maxdeg)))
    test_R = np.zeros(shape=(maxdeg, 1))
    train_R = np.zeros(shape=(maxdeg, 1))
    test_M = np.zeros(shape=(maxdeg, 1))
    train_M = np.zeros(shape=(maxdeg, 1))
    for i in range(maxdeg):
        t = train(i+1)
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


def task_a():
    deg = 5
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    print_cont(deg, err, "error")
    print_cont(deg, beta, "betas")
    print_cont(deg, np.nan_to_num(np.divide(err, beta), nan=0), "errorByBeta")
    errors = [test_M, train_M, test_R, train_R]
    labels = ["test MSE", "train MSE", "test R^2", "train R^2"]
    print_errors(deg, errors, labels, "errors")


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


def task_b():
    deg = MAX_DEG
    test_R, train_R, test_M, train_M, beta, var, err = train_degs(deg)
    errors = [test_M, train_M]
    labels = ["test MSE", "train MSE"]
    print_errors(deg, errors, labels, "errors_highdeg", True)


NOISE_LEVEL = 0.2
MAX_DEG = 25
RESOLUTION = .05
RANDOM_SEED = 1337


np.random.seed(RANDOM_SEED)
#plot_franke()
SCALE_DATA = False
task_a()
SCALE_DATA = True
task_b()
