#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    
    from pennylane.optimize import NesterovMomentumOptimizer,AdagradOptimizer,AdamOptimizer
    
    dev = qml.device("default.qubit", wires=2)
    
    def get_angles(x):

        beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
        beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
        beta2 = 2 * np.arcsin(
            np.sqrt(x[2] ** 2 + x[3] ** 2)
            / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
        )

        return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

    X = X_train[:, 0:3]
    X_T = X_test[:, 0:3]
    
    #pad the vectors to size 2^2 with constant values
    

    padding = 0.03 * np.ones((len(X), 1))
    X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
    X_pad = X_pad[:,0:4]

    padding = 0.03 * np.ones((len(X_T), 1))
    X_Tpad = np.c_[np.c_[X_T, padding], np.zeros((len(X_T), 1))]
    X_Tpad = X_Tpad[:,0:4]

    
    normalization = np.sqrt(np.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T

    normalization = np.sqrt(np.sum(X_Tpad ** 2, -1))
    X_Tnorm = (X_Tpad.T / normalization).T
    
    features = np.array([get_angles(x) for x in X_norm])
    features_T = np.array([get_angles(x) for x in X_Tnorm])
    
    
    Y = Y_train



    def statepreparation(a):
        qml.RY(a[0], wires=0)

        qml.CNOT(wires=[0, 1])
        qml.RY(a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[2], wires=1)

        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[4], wires=1)
        qml.PauliX(wires=0)

    def layer(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.CNOT(wires=[0, 1])


    @qml.qnode(dev)
    def circuit(weights, x):

        #statepreparation(np.array([get_bin_rep(num) for num in x]))
        statepreparation(x)

        for W in weights:
            layer(W)

        return qml.expval(qml.PauliZ(0)) #qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss

    def cost(var, X, Y):
        predictions = [variational_classifier(var, x) for x in X]
        return square_loss(Y, predictions)

    def variational_classifier(var, x):
        weights = var[0]
        bias = var[1]
        return circuit(weights, x) + bias
    
    def cost(var, X, Y):
        predictions = [variational_classifier(var, x) for x in X]
        return square_loss(Y, predictions)


    def accuracy(labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def predict(num):

        if num>0.33333:

            return 1

        elif  num< -0.333333:

            return -1

        else: 
            return 0





    
    
    np.random.seed(0)

    num_data = len(Y)
    num_train = int(0.95 * num_data)
    index = np.random.permutation(range(num_data))

    feats_train = features[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feats_val = features[index[num_train:]]
    Y_val = Y[index[num_train:]]

    # We need these later for plotting
    X_train = X[index[:num_train]]
    X_val = X[index[num_train:]]
    #Optimization
    #First we initialize the variables.

    num_qubits = 2
    num_layers = 10
    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
    #Again we optimize the cost. This may take a little patience.

    opt = AdagradOptimizer(0.07)
    batch_size = 10

    # train the variational classifier
    var = var_init
    for it in range(30):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(v, feats_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [predict(variational_classifier(var, f)) for f in feats_train]
       # predictions_val = [predict(variational_classifier(var, f)) for f in feats_val]

        # Compute accuracy on train and validation set
        #acc_train = accuracy(Y_train, predictions_train)
        #acc_val = accuracy(Y_val, predictions_val)
   
    for data in features_T:

         predictions.append(predict(variational_classifier(var, data)) )

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
