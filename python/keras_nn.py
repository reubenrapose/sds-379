import numpy as np
import keras.backend as K
import tensorflow as tf
import sys
from matplotlib import pyplot
import os

"""
This script is only included because it has evidence of attempts to use Tensorflow.
The code, in its current state, is incomplete. Tensorflow was tested a few times,
but ultimately deemed by the author to be inferior to PyTorch for custom loss functions.
"""

nodes_per_layer = 2
layers = 1
p = [-0.779439, -0.019717, -0.291465, -0.581873, 0.004982, 0.953471, -0.812328, -1.526382, 1.899841]

def main():
    # print(sys.__path__())
    print(os.getcwd())
    file_prefix = "function_approximation1"
    with open("outputs/"+file_prefix+".txt", 'w') as file:
        # Redirect stdout to the file
        # sys.stdout = file
        designed_nn(file_prefix)

    # Reset stdout to the default (console)
    sys.stdout = sys.__stdout__
    print("Complete")


def fitting_err(model, zpde, zic, zbc1, zbc2, p=p):
    zpde = tf.convert_to_tensor(zpde)
    zic = tf.convert_to_tensor(zic)
    zbc1 = tf.convert_to_tensor(zbc1)
    zbc2 = tf.convert_to_tensor(zbc2)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(zpde)
        with tf.GradientTape() as tape2:
            tape2.watch(zpde)
            predictions = model(zpde)
        first_deriv = tape2.gradient(predictions, zpde)
        ut = first_deriv[:,1]
        print("ut ad: ", ut)
    second_deriv = tape1.gradient(first_deriv, zpde)
    uxx = second_deriv[:,0]
    print("uxx ad: ", uxx)

    pde = ut - 1/np.pi * uxx
    # ic_term = tf.convert_to_tensor([
    #     get_fp(zici, p, tf.tanh) - tf.cast(tf.sin(np.pi*zici[0]), tf.float32)
    #                      for zici in zic])
    # bc_term1 = tf.convert_to_tensor([get_fp(zbc1i, p, tf.tanh) for zbc1i in zbc1])
    # bc_term2 = tf.convert_to_tensor([get_fp(zbc2i, p, tf.tanh) for zbc2i in zbc2])
    ic_term = tf.convert_to_tensor([
        model(zic) - tf.cast(tf.sin(np.pi * zici[0]), tf.float32)
        for zici in zic])
    bc_term1 = tf.convert_to_tensor([model(zbc1) for zbc1i in zbc1])
    bc_term2 = tf.convert_to_tensor([model(zbc2) for zbc2i in zbc2])

    pde = tf.cast(pde, tf.float32)
    ic_term = tf.cast(ic_term, tf.float32)
    bc_term1 = tf.cast(bc_term1, tf.float32)
    bc_term2 = tf.cast(bc_term2, tf.float32)

    output = K.sum(K.square(pde))/len(zpde) +\
             K.sum(K.square(ic_term))/len(zic) +\
             K.sum(K.square(bc_term1))/len(zbc1) +\
             K.sum(K.square(bc_term2))/len(zbc2)

    return output

def fitting_err_ode(model, z):
    z = tf.convert_to_tensor(z)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(z)
        with tf.GradientTape() as tape2:
            tape2.watch(z)
            modelode = model(z)
        first_deriv = tape2.gradient(modelode, z)
    second_deriv = tape1.gradient(first_deriv, z)
    uxx = tf.cast(second_deriv, tf.float32)

    ode = uxx + modelode
    model0 = model[0]
    dmodel0 = first_deriv[0]

    # output = (K.sum(K.square(ode))) / len(z)

    return output

def designed_nn(file_prefix): # https://machinelearningmastery.com/neural-networks-are-function-approximators/
    # example of fitting a neural net on x vs x^2
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense
    # define the dataset

    # zpde = np.array([[np.random.rand(), np.random.rand()*np.pi/2] for _ in range(25)])
    # zic = np.array([[np.random.rand(), 0] for _ in range(10)])
    # zbc1 = np.array([[0, np.random.rand()] for _ in range(10)])
    # zbc2 = np.array([[np.pi/2, np.random.rand()] for _ in range(10)])
    #
    # # reshape arrays into into rows and cols
    # zpde = zpde.reshape((zpde.shape[0], zpde.shape[1]))
    # zic = zic.reshape(zic.shape[0], zic.shape[1])
    # zbc1 = zbc1.reshape(zbc1.shape[0], zbc1.shape[1])
    # zbc2 = zbc2.reshape(zbc2.shape[0], zbc2.shape[1])

    # z = np.concatenate((zpde, zbc1, zbc2), axis=0)

    z = np.random.rand(49) * 6
    z = np.insert(z, 0, 0)

    nodes_per_layer = 8
    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=1, activation='tanh'))
    model.add(Dense(nodes_per_layer, activation='tanh'))
    model.add(Dense(1))
    # define the loss function and optimization algorithm

    print(f"{8}x{2} Network to soln to x_tt + x = 0, x(0) = 1, x'(0)=2:\n\n")
          # f"ut - 1/pi * uxx = 0, u(x,0) = sin(pi*x), u(0,t)=u(1,t)=0\n\n")

    # model.compile(loss=lambda y_true, y_pred: fitting_err(model, zpde, zic, zbc1, zbc2),
    #               optimizer='adam')
    model.compile(loss=lambda y_true, y_pred: fitting_err_ode(model, z),
                  optimizer="adam")

    # ft the model on the training dataset
    # model.fit(z, y, epochs=800000, verbose=2)

    # z0pde, z1pde = np.array([zpde[i][0] for i in range(len(zpde))]), np.array([zpde[i][1] for i in range(len(zpde))])
    # z0ic, z1ic = np.array([zic[i][0] for i in range(len(zic))]), np.array([zic[i][1] for i in range(len(zic))])
    # z0bc1, z1bc1 = np.array([zbc1[i][0] for i in range(len(zbc1))]), np.array([zbc1[i][1] for i in range(len(zbc1))])
    # z0bc2, z1bc2 = np.array([zbc2[i][0] for i in range(len(zbc2))]), np.array([zbc2[i][1] for i in range(len(zbc2))])
    #
    # z0 = np.concatenate((z0pde, z0ic, z0bc1, z0bc2))
    # z1 = np.concatenate((z1pde, z1ic, z1bc1, z1bc2))
    #
    # z0, z1 = np.meshgrid(z0, z1)
    #
    #
    # z0_reshaped = z0.reshape(-1, 1)
    # z1_reshaped = z1.reshape(-1, 1)
    # # Combine z0_reshaped and z1_reshaped into a single input array
    # z_input = np.concatenate((z0_reshaped, z1_reshaped), axis=1)
    # yhat = model.predict(z_input)
    # yhat = yhat.reshape(z0.shape)
    # fz = np.exp(-np.pi*z1) * np.sin(np.pi*z0)


    # Predict using the model
    # model.save_weights("outputs/model/" + file_prefix + ".weights.h5")
    model.load_weights("outputs/model/function_approximation1.weights.h5")

    # Reshape yhat back to the shape of z0_mesh and z1_mesh

    z = np.arange(1.5, 3, .01)
    fz = np.cos(z) + 2*np.sin(z)
    yhat = model.predict(z)

    print(list(fz))
    print(list(np.concatenate(yhat)))

    pyplot.scatter(z, fz, label="Actual")
    pyplot.scatter(z, yhat, label="Predicted")
    pyplot.title("Approximating soln to x_tt + x = 0, x(0) = 1, x'(0) = 2")
    pyplot.xlabel("x")
    pyplot.ylabel("f(x) vs N(x)")
    pyplot.legend()
    pyplot.show()


    # print(f"Shapes (x, t, fz, yhat): {z0.shape}, {z1.shape}, {fz.shape}, {yhat.shape}\n")

def plot(z0, z1, fz, yhat):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(z0, z1, fz, c="r", label="Actual")
    ax.scatter(z0, z1, yhat, c="b", label="Predicted")
    ax.set_title("u(x,t) vs N(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()

    fig2 = pyplot.figure()
    ax2 = fig2.add_subplot(121, projection="3d")
    ax2.plot_surface(z0, z1, fz, cmap='viridis')
    ax2.set_title("u(x,t)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u")
    ax3 = fig2.add_subplot(122, projection="3d")
    ax3.plot_surface(z0, z1, yhat, cmap='inferno')
    ax3.set_title("N(x,t)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_zlabel("u")
    pyplot.show()

def designed_nn_test_weight(z_input):
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib
    from matplotlib import pyplot

    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='tanh'))
    # model.add(Dense(8, activation='tanh'))
    model.add(Dense(1))
    # define the loss function and optimization algorithm

    model.compile(optimizer='adam')

    model.load_weights("outputs/model/test.weights.h5")

    # Predict using the model
    yhat = model.predict(z_input)

    print("\n\nModel 2:\n\nShape: ", yhat.shape, "\n\n", yhat)

def get_fp(zi, p, actfunction):
    output = 0
    pi = 0
    # construct first layer
    prev_nodes = []
    for ni in range(nodes_per_layer):
        weighted = 0
        for zii in range(len(zi)):
            weighted += p[pi] * zi[zii]
            pi += 1
        node = actfunction(weighted + p[pi])
        prev_nodes.append(node)
        pi += 1

    for li in range(layers-1):
        new_nodes = []
        for nni in range(nodes_per_layer):
            weighted = 0
            for pni in range(nodes_per_layer):
                weighted += p[pi] * prev_nodes[pni]
                pi += 1
            node = actfunction(weighted + p[pi])
            new_nodes.append(node)
            pi += 1
        prev_nodes = new_nodes

    for ni in range(nodes_per_layer):
        output += p[pi] * prev_nodes[ni]
        pi += 1

    return output + p[pi]


def fitting_err_fdm(zpde, zic, zbc1, zbc2, p = p):
    output = 0
    h = 1e-3
    # zpde = tf.convert_to_tensor(zpde)

    dfpdxx = ([
        (get_fp([zpde[i][0]+h, zpde[i][1]], p, tf.tanh)
        - 2*get_fp(zpde[i], p, tf.tanh)
        + get_fp([zpde[i][0]-h, zpde[i][1]], p, tf.tanh))/(h**2)
        for i in range(len(zpde))
    ])
    print("uxx fdm: ", dfpdxx)

    dfpdt = ([
        (get_fp([zpde[i][0], zpde[i][1]+h], p, tf.tanh)
         - get_fp([zpde[i][0], zpde[i][1]-h], p, tf.tanh))/(2*h)
        for i in range(len(zpde))
    ])
    print("ut fdm: ", dfpdt)

    pde = dfpdt - 1/np.pi * dfpdxx
    ic_term = tf.convert_to_tensor([get_fp(zic[i], p, tf.tanh) - tf.sin(np.pi*zic[i][0])
                                    for i in range(len(zic))])
    bc_term1 = tf.convert_to_tensor([get_fp(zbc1[i], p, tf.tanh)
                                     for i in range(len(zbc1))])
    bc_term2 = tf.convert_to_tensor([get_fp(zbc2[i], p, tf.tanh)
                                     for i in range(len(zbc2))])

    output = K.sum(K.square(pde))/len(zpde) +\
             K.sum(K.square(ic_term))/len(zic) +\
             K.sum(K.square(bc_term1))/len(zbc1) +\
             K.sum(K.square(bc_term2))/len(zbc2)
    return output

if __name__ == "__main__":
    main()