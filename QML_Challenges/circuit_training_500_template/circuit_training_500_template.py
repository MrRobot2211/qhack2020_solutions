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
    from pennylane.templates.embeddings import AmplitudeEmbedding
    from qml.templates.layers import RandomLayers


    dev = qml.device("default.qubit", wires=4)
    
    
    
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
    
    features = X
    features_T= X_T
    #pad the vectors to size 2^2 with constant values
    
    Y = Y_train



    def statepreparation(a):
        AmplitudeEmbedding(a,wires=range(4),pad_with=0.3,normalize=True)
       # qml.Hadamard(wires=2)
        
        #qml.Hadamard(wires=3)



    @qml.qnode(dev)
    def circuit(weights, x):

        #statepreparation(np.array([get_bin_rep(num) for num in x]))
        statepreparation(x)

    #     for i,W in eumerate(weights):

    #         if i < len(W)-1: 
    #             layer(W)

    #         else:

        RandomLayers(weights,wires = range(3))

        return qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2)) #qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))



    def variational_classifier(var, x):

        weights = var[0]
        bias = var[1]
        return circuit(weights, x) + bias



    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss
    
   
    
    def bce_loss(labels, predictions):
        
        classes_encoder={-1:0,0:1,1:2}
        
        
        loss = 0
        for l, p in zip(labels, predictions):

            preds =  np.exp(p)/sum(np.exp(p))
        
            loss = loss -np.log(preds[classes_encoder[l.numpy()]])

     
    
        loss = loss / len(labels)
        return loss
    
    def cost(var, X, Y):
        predictions = [variational_classifier(var, x) for x in X]
        return bce_loss(Y, predictions )


    def predict(num):
        classes=[-1,0,1]
        return classes[np.argmax(num)]
        

    
    def accuracy(labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss
    
    np.random.seed(0)
    num_data = len(Y)
    num_train = int(0.95 * num_data)
    EPOCHS=15
    
    
    
    index = np.random.permutation(range(num_data))

    feats_train = features[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feats_val = features[index[num_train:]]
    Y_val = Y[index[num_train:]]

    # We need these later for plotting
   # X_train = X[index[:num_train]]
   # X_val = X[index[num_train:]]
    #Optimization
    #First we initialize the variables.

    num_qubits = 3
    num_layers = 10
    #var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
    var_init = (np.random.randn(num_layers, 15), np.random.randn(3))
    
    
    #Again we optimize the cost. This may take a little patience.

    opt = NesterovMomentumOptimizer(0.1)
    batch_size = 15
    
    # train the variational classifier
    var = var_init
    for it in range(EPOCHS):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        
        
        var = opt.step(lambda v: cost(v, feats_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [predict(variational_classifier(var, f)) for f in feats_train]
        predictions_val = [predict(variational_classifier(var, f)) for f in feats_val]

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)
        
        print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost(var, features, Y), acc_train, acc_val)
    )
         
   
    for data in features_T:

         predictions.append(predict(variational_classifier(var, data)) )
            
      

    # QHACK #

    return array_to_concatenated_string(predictions)

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
