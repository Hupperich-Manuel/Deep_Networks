import numpy as np
import matplotlib.pyplot as plt

def initialize(layer_dim):

    L = len(layer_dim)
    parameters = {}

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.rand(layer_dim[l], layer_dim[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dim[l], 1))

    return parameters

def initialize_adam(parameters):

    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(1, L+1):
        v['dW'+str(l)] = np.zeros((parameters["W"+str(l)].shape))
        v['db' + str(l)] = np.zeros((parameters["b" + str(l)].shape))

        s["dW"+str(l)] = np.zeros((parameters["W"+str(l)].shape))
        s['db' + str(l)] = np.zeros((parameters["b" + str(l)].shape))

    return v, s

def update_params_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon):

    L = len(parameters)//2
    v_new = {}
    s_new = {}

    for l in range(1, L+1):
        v['dW' + str(l)] = beta1*v['dW' + str(l)] + (1-beta1)*grads['dW' + str(l)]
        v['db' + str(l)] = beta1*v['db' + str(l)] + (1-beta1)*grads['db' + str(l)]
        ''''''
        s["dW" + str(l)] = beta2*s['dW'+str(l)] + (1-beta2)*grads['dW' + str(l)]**2
        s["db" + str(l)] = beta2*s['db'+str(l)] + (1-beta2)*grads['db' + str(l)]**2
        ''''''
        v_new['dW' + str(l)] = v['dW' + str(l)]/(beta1**t)
        v_new['db' + str(l)] = v['db' + str(l)]/(beta1**t)
        ''''''
        s_new['dW' + str(l)] = s['dW' + str(l)]/(beta2 ** t)
        s_new['db' + str(l)] = s['db' + str(l)]/(beta2 ** t)
        ''''''
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate*(v_new["dW"+str(l)]/(np.sqrt(s_new['dW' + str(l)] + epsilon)))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate*(v_new["db" + str(l)]/(np.sqrt(s_new['db' + str(l)] + epsilon)))

    return parameters

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


