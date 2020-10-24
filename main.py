import sys

from sklearn.preprocessing import scale
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers

from regression import task_a, create_grid, RESOLUTION, franke, NOISE_LEVEL, mse_error, r2_error, \
    print_errors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from network import LayerDense, Costfunctions, ActivationFunctions, Metrics, Network
import numpy as np

from sklearn import datasets

#task_a()

from keras import backend as K


#https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
def coeff_determination(y_true, y_pred):
    r = (K.square(y_true - y_pred)) / K.sum(K.square(y_true - K.mean(y_pred)))
    return 1 - K.sum(r, axis=0)



def create_terrain_data():
    split_size = 0.8
    x, y = create_grid(RESOLUTION)
    z = np.expand_dims(franke(x, y, NOISE_LEVEL).flatten(), axis=1)
    feature_mat = np.zeros(shape=(x.size, 2))
    feature_mat[:, 0] = x.flatten()
    feature_mat[:, 1] = y.flatten()
    x_train, x_test, y_train, y_test = train_test_split(feature_mat, z, train_size=split_size)
    return x_train, x_test, y_train, y_test


def print_metrik_by_network_and_epoch(num_epochs, cost, task, name, metric_name):
    fig = plt.figure(figsize=(6, 6), dpi=300)
    if len(cost) == 2:
        plt.plot(np.linspace(0, num_epochs, num_epochs+1), cost[0], label="train", linestyle="--")
        plt.plot(np.linspace(0, num_epochs, num_epochs+1), cost[1], label="test", linestyle="-")
    else:
        plt.plot(np.linspace(0, num_epochs, num_epochs+1), cost, linestyle="--")
    plt.legend()
    plt.grid()
    ax = fig.gca()
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric_name)
    plt.yscale('log')
    fig.tight_layout()
    fig.savefig("images/" + task + "/" + name + ".png", dpi=300)
    plt.close()


def create_network_array(n_in, n_out, learning_rate, cf, af, start, stop):
    networks = []
    for i in range(start, stop+1):
        li = LayerDense(n_in, 2**i, "ldi"+str(i), learning_rate, cf, af)
        lm = LayerDense(2**i, 2**i, "ldm"+str(i), learning_rate, cf, af)
        lo = LayerDense(2**i, n_out, "lo"+str(i), learning_rate, cf, ActivationFunctions.relu)
        networks.append(Network([li, lo], str(i)+"short"))
        networks.append(Network([li, lm, lo], str(i)+"long"))
    return networks


def create_keras_network_array(n_in, n_out, learning_rate, cf, af, start, stop, task):
    models = []
    for i in range(start, stop+1):
        model1 = keras.Sequential(name=str(i)+"short")
        model1.add(layers.Dense(n_in, input_shape=(n_in,), activation=af, kernel_initializer='he_uniform'))
        model1.add(layers.Dense(2 ** i, activation=af, kernel_initializer='he_uniform'))
        model1.add(layers.Dense(n_out, activation='relu', kernel_initializer='he_uniform'))
        plot_model(model1, to_file='images/b/'+model1.name+'.png', show_shapes=True, show_layer_names=True)
        #model1.summary()
        model1.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                       loss=cf, metrics=['MeanSquaredError', coeff_determination])
        model2 = keras.Sequential(name=str(i)+"long")
        model2.add(layers.Dense(n_in, input_shape=(n_in,), activation=af, kernel_initializer='he_uniform'))
        model2.add(layers.Dense(2 ** i, activation=af))
        model2.add(layers.Dense(2 ** i, activation=af))
        model2.add(layers.Dense(n_out, activation='relu'))
        plot_model(model2, to_file='images/'+task+'/'+model2.name+'.png', show_shapes=True, show_layer_names=True)
        #model2.summary()
        model2.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                       loss=cf, metrics=['MeanSquaredError', coeff_determination])
        models.append(model1)
        models.append(model2)
        del model1
        del model2
    return models


def task_b_own():
    learning_rate = 0.1
    full_network_test(Costfunctions.mse, ActivationFunctions.sigmoid, learning_rate, "b")


def full_network_test(cf, af, learning_rate, task, epochs=50):
    x_train, x_test, y_train, y_test = create_terrain_data()
    start = 3
    stop = 8
    n = stop-start+1
    networks = create_network_array(x_train.shape[1], y_train.shape[1], learning_rate, cf, af,
                                    start, stop)
    batch_size = 50
    mse_errors = []
    r2_scores = []
    for network in networks:
        network.train(x_train.T, y_train.T, epochs, batch_size, x_test.T, y_test.T)
        mse_test = mse_error(y_test, network.predict(x_test.T).T)[0]
        mse_train = mse_error(y_train, network.predict(x_train.T).T)[0]
        r2_test = r2_error(y_test, network.predict(x_test.T).T)[0]
        r2_train = r2_error(y_train, network.predict(x_train.T).T)[0]
        print("Cost function after initialization, after first epoch and last epoch ")
        print(network.train_C[0], network.train_C[1], network.train_C[-1])
        print_metrik_by_network_and_epoch(epochs, [network.train_C, network.test_C], task,
                                        "own_" + network.name + "_" + af.__name__)
        mse_errors.append([mse_test, mse_train])
        r2_scores.append([r2_test, r2_train])
    mse_errors = np.array(mse_errors).reshape(n, 4).T
    r2_scores = np.array(r2_scores).reshape(n, 4).T
    print_errors(np.linspace(start, stop, n), mse_errors, ["short_test", "short_train", "long_test", "long_train"], "own_mse_"+af.__name__, logy=True,
                 xlabel="ln_2 of neurons per layer", ylabel="mse", task=task)
    print_errors(np.linspace(start, stop, n), r2_scores, ["short_test", "short_train", "long_test", "long_train"], "own_r_squared_"+af.__name__, logy=True,
                 xlabel="ln_2 of neurons per layer", ylabel="R^2 value", task=task)


def task_b_keras():
    x_train, x_test, y_train, y_test = create_terrain_data()
    learning_rate = 0.001
    start = 3
    stop = 8
    n = stop-start+1
    epochs = 50
    batch_size = 50
    models = create_keras_network_array(x_train.shape[1], y_train.shape[1], learning_rate,
                                        'MeanSquaredError', 'sigmoid', start, stop, "b")
    mse_errors = []
    r2_scores = []
    for model in models:
        print(model.name)
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(x_test, y_test), verbose=0)
        mse_errors.append([model.evaluate(x_test, y_test)[1], model.evaluate(x_train, y_train)[1]])
        r2_scores.append([model.evaluate(x_test, y_test)[2], model.evaluate(x_train, y_train)[2]])
        print_metrik_by_network_and_epoch(epochs-1, [history.history['loss'], history.history['val_loss']], "b", "keras_"+model.name)

    mse_errors = np.array(mse_errors).reshape(n, 4).T
    r2_scores = np.array(r2_scores).reshape(n, 4).T
    print_errors(np.linspace(start, stop, n), mse_errors, ["short_test", "short_train", "long_test", "long_train"], "keras_mse", logy=True,
                 xlabel="ln_2 of neurons per layer", ylabel="mse", task="b")
    print_errors(np.linspace(start, stop, n), r2_scores, ["short_test", "short_train", "long_test", "long_train"], "keras_r_squared", logy=True,
                 xlabel="ln_2 of neurons per layer", ylabel="R^2 value", task="b")


def task_c():
    learning_rate = 0.1
    full_network_test(Costfunctions.mse, ActivationFunctions.relu, learning_rate, "c", epochs=100)
    full_network_test(Costfunctions.mse, ActivationFunctions.leaky_relu, learning_rate, "c", epochs=100)
    full_network_test(Costfunctions.mse, ActivationFunctions.elu, learning_rate, "c", epochs=100)


def prepare_digit_data():
    digits = datasets.load_digits()
    x = digits.data
    t = digits.target
    n_out = 10
    y = np.zeros(shape=(x.shape[0], n_out))
    y[np.arange(t.size), t] = 1
    split_size = 0.8
    x = (x - np.max(x) / 2) / (np.max(x) / 2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_size)
    return x_train, x_test, y_train, y_test


def pred_to_class(y):
    r = np.argmax(y, axis=0)
    return r


def task_d():
    x_train, x_test, y_train, y_test = prepare_digit_data()
    n_out = 10
    n_in = x_train.shape[1]
    n_middle = 256
    epochs = 250
    batch_size = 100
    learning_rate = 0.0001
    l_in = LayerDense(n_in, n_middle, "lin", learning_rate, Costfunctions.mse,
                      ActivationFunctions.elu)
    l_mi = LayerDense(n_middle, n_middle, "lmi2", learning_rate, Costfunctions.mse,
                      ActivationFunctions.sigmoid)
    l_ou = LayerDense(n_middle, n_out, "lou", learning_rate, Costfunctions.mse,
                      ActivationFunctions.softmax)
    network = Network([l_in, l_mi, l_ou], "mnist", [Metrics.mse, Metrics.accuracy])
    network.train(x_train.T, y_train.T, epochs, batch_size, x_test.T, y_test.T)
    print("Metric after initialization, after first epoch and last epoch ")
    print(network.get_train_met()[0,:], network.get_train_met()[1,:], network.get_train_met()[-1,:])
    print_metrik_by_network_and_epoch(epochs, [network.get_train_met()[:,0], network.get_test_met()[:,0]], "d", "mse_own_"+"_".join([network.name,"epochs",str(epochs),"lr",str(learning_rate),"nodes",str(n_middle),"batch_size",str(batch_size)]), "entropy")
    print_metrik_by_network_and_epoch(epochs, [network.get_train_met()[:,1], network.get_test_met()[:,1]], "d", "acc_own_"+"_".join([network.name,"epochs",str(epochs),"lr",str(learning_rate),"nodes",str(n_middle),"batch_size",str(batch_size)]), "accuracy")



task_d()

"""
x_train, x_test, y_train, y_test = prepare_digit_data()
n_out = 10
n_in = x_train.shape[1]
n_middle = 256
learning_rate = 0.0001
epochs = 250
batch_size = 100

l_in = LayerDense(n_in, n_middle, "lin", learning_rate, Costfunctions.mse,
                  ActivationFunctions.elu)
l_mi = LayerDense(n_middle, n_middle, "lmi", learning_rate, Costfunctions.mse,
                  ActivationFunctions.elu)
l_mi2 = LayerDense(n_middle, n_middle, "lmi2", learning_rate, Costfunctions.mse,
                  ActivationFunctions.sigmoid)
l_ou = LayerDense(n_middle, n_out, "lou", learning_rate, Costfunctions.mse,
                  ActivationFunctions.softmax)
network = Network([l_in, l_mi2, l_ou], "mnist", [Metrics.ce, Metrics.accuracy, Metrics.ce_grad])
network.train(x_train.T, y_train.T, epochs, batch_size, x_test.T, y_test.T)
network.layers[-1].cf=Costfunctions.cross_entropy
network.layers[-1].cf_grad=Costfunctions.cross_entropy_grad
network.train(x_train.T, y_train.T, epochs, batch_size, x_test.T, y_test.T)

print("Metric after initialization, after first epoch and last epoch ")
print(network.get_train_met()[0,:], network.get_train_met()[1,:], network.get_train_met()[-1,:])
print_metrik_by_network_and_epoch(epochs*2+1, [network.get_train_met()[:,0], network.get_test_met()[:,0]], "d", "cre_own_"+"_".join([network.name,"epochs",str(epochs),"lr",str(learning_rate),"nodes",str(n_middle),"batch_size",str(batch_size)]), "entropy")
print_metrik_by_network_and_epoch(epochs*2+1, [network.get_train_met()[:,1], network.get_test_met()[:,1]], "d", "acc_own_"+"_".join([network.name,"epochs",str(epochs),"lr",str(learning_rate),"nodes",str(n_middle),"batch_size",str(batch_size)]), "accuracy")
print_metrik_by_network_and_epoch(epochs*2+1, [network.get_train_met()[:,2], network.get_test_met()[:,2]], "d", "ceg_own_"+"_".join([network.name,"epochs",str(epochs),"lr",str(learning_rate),"nodes",str(n_middle),"batch_size",str(batch_size)]), "ent grad")
"""

#task_b_own()
#task_b_keras()
#task_c()
#task_d()
