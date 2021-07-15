import numpy as np
import pandas as pd
from Deep import Adam_Opt
import sklearn.datasets
import matplotlib.pyplot as plt
from Deep import Datasets
from Deep import Learning_rate


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    #plt.show()

    return train_X, train_Y


X, Y = load_dataset()

df = Datasets.ratios_stocks()
train_X, train_Y, test_X, test_Y = np.array(df[df.columns[0:-1]][:125].T), np.array(df[df.columns[-1]][:125]), np.array(df[df.columns[0:-1]][125:].T), np.array(df[df.columns[-1]][125:])

train_Y = np.expand_dims(train_Y, axis=0)
test_Y = np.expand_dims(test_Y, axis=0)

#print(train_X.shape)
#print(train_Y.shape)
#print(test_X.shape)
#print(test_Y.shape)


def relu(x):
    """Relu function"""
    return np.maximum(0, x)
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def initialize_parameters(layers_dimension):
    """Parameters setting
    n_x size of the input layer
    n_h size of the hidden layer
    n_y size of the output layer
    layer_dimesnion = [12288, 20, 7, 5, 1]
                      [X.shape[0], 7, 5, Y.shape[1]]"""
    parameters = {}
    L = len(layers_dimension)
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layers_dimension[i], layers_dimension[i-1])*np.sqrt(2/layers_dimension[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dimension[i], 1))

    return parameters


def forward_propagation(X, parameters):
    """jnejvn"""

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def compute_cost(a3, y):
    logprobabilities = np.multiply(-np.log(a3),y) + np.multiply(-np.log(1 - a3), 1 - y)
    cost = np.sum(logprobabilities)
    return cost

def initialize_velocity(parameters):

    v = {}
    L = len(paramters) // 2
    for l in range(1, L+1):
        v["dW"+str(l)] = np.random.randn(parameters["W"+str(l)].shape)
        v["db"+str(l)] = np.zeros((parameters["b"+str(l)].shape))
    return v

def backward_prop(cache, y, x):

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    m = x.shape[1]
    dZ3 = 1/m*(A3-y)
    dW3 = np.dot(dZ3, A3.T)
    db3 = np.sum(dZ3,  axis=1, keepdims=True)

    dA2 = np.dot(dW3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(dW2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0)) #np.multiply because it is an elementwise multiplication
    dW1 = np.dot(dZ1, x.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,"dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def update_parameters_with_gd(gradients, parameters, learning_rate):
    L = len(parameters)//2

    for l in range(1, L):

        parameters["dW"+str(l)] = parameters["W"+str(l)] - learning_rate*gradients["dW"+str(l)]
        parameters["dW"+str(l)] = parameters["W"+str(l)] - learning_rate*gradients["dW"+str(l)]

    return parameters



def model(X, Y, layers_dims,learning_rate=.0007, num_itinerations=5000,beta1=0.9, beta2=0.99, t=2, epsilon= 1e-2, print_cost = True, decay=None, decay_rate=1):
    costs = []
    parameters = initialize_parameters(layers_dims)
    v, s = Adam_Opt.initialize_adam(parameters)
    learning_rate0 = learning_rate
    m = X.shape[1]

    for i in range(0, num_itinerations):

        a3, cache = forward_propagation(X, parameters)

        cost = compute_cost(a3, Y)

        grads = backward_prop(cache, Y, X)

        parameters = Adam_Opt.update_params_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate0, epsilon)

        cost_avg = cost/m

        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)
            # Print the cost every 1000 itineraion
        if print_cost and i%1000 == 0:
            print("Cost after itineraion %i: %f"%(i, cost_avg))
            if decay:
                print("learning rate after itineraion %i: %f"%(i, learning_rate))
        if print_cost and i%100 == 0:
            costs.append(cost_avg)

    return parameters, costs



def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    p = np.zeros((1, m), dtype='int64')

    # Forward propagation
    a3, cache = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results

    print ("predictions: " + str(p[0,:]))
    print ("true labels: " + str(y[0,:]))
    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

    return p


# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters, costs = model(train_X, train_Y, layers_dims, learning_rate = 0.01, decay=Learning_rate.update_lr)
#print(parameters["W1"])

# Predict
predictions = predict(train_X, train_Y, parameters)
predictions_test = predict(test_X, test_Y, parameters)


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


