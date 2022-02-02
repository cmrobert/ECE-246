import numpy as np
import mnist
import copy
import random
import argparse
import pdb


# Load MNIST data
mndata = mnist.MNIST('data')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing()

X_train = np.asarray(X_train) / 255
X_test = np.asarray(X_test) / 255

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

# some statistics about the dataset
num_train = np.shape(X_train)[0]
num_features = np.shape(X_train)[1]
num_test = np.shape(X_test)[0]
num_classes = len(set(Y_train))

# Added this code (MK)
parser = argparse.ArgumentParser(description='hyperparameter sweeps')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--batch_size', default=5, type=int)
args = parser.parse_args()

# ------------- Global parameters-----------------------
num_hidden = 200  # number of nodes in hidden layer
num_epochs = 20  # number of epochs (for each worker)
num_workers = args.num_workers  # number of worker nodes
num_sample_worker = int(num_train / num_workers)  # number of samples per worker
num_test_workers = num_workers
num_sample_worker_test = int(num_test / num_test_workers)
batch_size = args.batch_size  # bath size for training. keep batch_size*num_workers constant for centralized and decentralized cases for a fair comparison
lr = 0.2  # initial learning rate
decay_factor = 0.99  # decay coefficient for learning rate
# ------------------------------------------------------


# Uniformly distribute data to the workers
def uniformSampling():
    train_datasets, test_datasets = [], []
    #######################################
    # YOUR CODE HERE:
    perm = np.random.permutation(num_train)  # randomize indices
    # Create length num_workers list, each item containing a tuple of (X, Y) samples of length num_sample_worker
    train_datasets = [(X_train[perm[worker * num_sample_worker:(worker + 1) * num_sample_worker]],
                       Y_train[perm[worker * num_sample_worker:(worker + 1) * num_sample_worker]])
                      for worker in range(num_workers)]
    # repeat for creating test datasets
    perm = np.random.permutation(num_test)
    test_datasets = [(X_test[perm[worker * num_sample_worker_test:(worker + 1) * num_sample_worker_test]],
                      Y_test[perm[worker * num_sample_worker_test:(worker + 1) * num_sample_worker_test]])
                     for worker in range(num_test_workers)]
    # print('train_datasets', len(train_datasets[0][0]))
    # CODE ENDS HERE
    #######################################
    return train_datasets, test_datasets


# Randomly initialize the parameters of the neural network
def initialize(num_inputs, num_classes, num_hidden):
    """initialize the parameters"""
    # num_inputs = 28*28 = 784
    # num_classes = 10
    # num_hidden is an input parameter
    params = {
        "W1": np.random.randn(num_hidden, num_inputs) * np.sqrt(1. / num_inputs),
        "b1": np.zeros((num_hidden, 1)) * np.sqrt(1. / num_inputs),
        "W2": np.random.randn(num_classes, num_hidden) * np.sqrt(1. / num_hidden),
        "b2": np.zeros((num_classes, 1)) * np.sqrt(1. / num_hidden)
    }
    return params


# ReLU non-linearlity
def ReLU(z):
    """
    ReLU activation function.
    inputs: z
    outputs: max(z,0)
    """
    r = z.clip(min=0)
    return r


def oneHotEncode(Y):
    E = np.zeros((num_classes, np.shape(np.asarray(Y))[0]))  # (10*num_batch)
    for i in range(np.shape(E)[1]):
        E[Y[i]][i] = 1
    return E

# Compute the cross entropy loss given estimate (Y_hat) and true labels (Y)


def compute_loss(Y, Y_hat):
    """
    compute loss function
    """
    E = oneHotEncode(Y)
    L_sum = np.sum(np.multiply(E, np.log(Y_hat)))
    m = Y.shape[0]
    L = (-1 * L_sum) / m
    return L


# Routine to perform forward pass on the neural network
def feed_forward(X, params):
    """
    feed forward network: 2 - layer neural net

    inputs:
        params: dictionay a dictionary contains all the weights and biases

    return:
        cache: dictionay a dictionary contains all the fully connected units and activations
    """
    cache = {}

    # The NN forward pass is as follows: Data -linear-> Z1 -(ReLU)-> A1 -(lienar)-> Z2 -(softmax)-> output

    # Update to perform: Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X.transpose()) + params["b1"]

    # Update to perform: A1 = ReLU(Z1)
    cache["A1"] = ReLU(cache["Z1"])

    # Update to perform: Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # Update to perform: A2 = softmax(Z2)
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache

# Routine to perform backward pass to calculate gradients


def back_propagate(X, Y, params, cache, m_batch):
    """
    back propagation

    inputs:
        params: dictionay a dictionary contains all the weights and biases
        cache: dictionay a dictionary contains all the fully connected units and activations

    return:
        grads: dictionay a dictionary contains the gradients of corresponding weights and biases
    """
    # error at last layer
    E = oneHotEncode(Y)
    dZ2 = cache["A2"] - E

    # gradients at last layer (Py2 need 1. to transform to float)
    dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # back propgate through first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dA1_copy = copy.deepcopy(dA1)
    dA1_copy[cache["Z1"] < 0] = 0
    dZ1 = dA1_copy

    # gradients at first layer (Py2 need 1. to transform to float)
    dW1 = (1. / m_batch) * np.matmul(dZ1, X)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# Routine to evaluate the accuracy and loss of the parameters on a given batch (x_data,y_data)
def eval(params, x_data, y_data):
    """ implement the evaluation function
    input: param -- parameters dictionary
           x_data -- x_train or x_test (size, 784)
           y_data -- y_train or y_test (size,)
    output: loss and accuracy
    """
    cache = feed_forward(x_data, params)
    loss = compute_loss(y_data, cache["A2"])
    result = np.argmax(np.array(cache["A2"]).T, axis=1)
    accuracy = sum(result == y_data) / float(len(y_data))
    return loss, accuracy


# Routine to perform a forward pass and then a back progpagate (back pass) to compute gradients
def mini_batch_gradient(params, x_batch, y_batch):
    """implement the function to compute the mini batch gradient
    input: params -- parameters dictionary
           x_batch -- a batch of x (size, 784)
           y_batch -- a batch of y (size,)
    output: gradients of the parameters
    """
    batch_size = x_batch.shape[0]
    cache = feed_forward(x_batch, params)
    grads = back_propagate(x_batch, y_batch, params, cache, batch_size)
    return grads


# Distributed training scheme
def trainDistributed(params, hyp, train_datasets, test_datasets):
    num_epochs = hyp['num_epochs']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list = [], [], [], []
    #######################################
    # YOUR CODE HERE:
    XYtrain, XYtest = train_datasets, test_datasets  # aliases
    for epoch in range(num_epochs):  # i.e. for each round
        worker_params = [copy.deepcopy(params) for worker in range(num_workers)]  # copy params for in-place operations
        for worker in range(num_workers):
            # start ClientUpdate
            num_batches = len(XYtrain[worker][0]) // batch_size  # i.e. number of batches for each worker
            perm = np.random.permutation(len(XYtrain[worker][0]))  # randomize batches within each worker
            num_local_epochs = 1  # number of local epochs to train for; use 1 for comparability
            for local_epoch in range(num_local_epochs):
                for batch in range(num_batches):
                    inds = perm[batch * batch_size:(batch + 1) * batch_size]  # get inds of perm to use
                    x_batch, y_batch = XYtrain[worker][0][inds], XYtrain[worker][1][inds]  # get batch x and y
                    batch_grads = mini_batch_gradient(worker_params[worker], x_batch, y_batch)  # get gradients for batch x and y
                    for param_name in list(params):
                        worker_params[worker][param_name] -= learning_rate * batch_grads['d' + param_name]  # update each param
                        # ^update 'W1' using 'dW1', etc.
            # end ClientUpdate

        # average param across all workers. n_k is the same for all workers
        for param_name in list(params):
            params[param_name] = np.mean([worker_params[worker][param_name] for worker in range(num_workers)], axis=0)

        # Concatenate x and y for centralized evaluation
        train_all = [None] * 2
        train_all[0] = np.concatenate([train_datasets[worker][0] for worker in range(num_workers)])
        train_all[1] = np.concatenate([train_datasets[worker][1] for worker in range(num_workers)])
        test_all = [None] * 2
        test_all[0] = np.concatenate([test_datasets[worker][0] for worker in range(num_workers)])
        test_all[1] = np.concatenate([test_datasets[worker][1] for worker in range(num_workers)])

        train_loss, train_accu = eval(params, train_all[0], train_all[1])
        test_loss, test_accu = eval(params, test_all[0], test_all[1])
        epoch_train_loss_list.append(train_loss)
        epoch_train_accu_list.append(train_accu)
        epoch_test_loss_list.append(test_loss)
        epoch_test_accu_list.append(test_accu)
        message = 'Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch + 1, train_loss, train_accu, test_loss, test_accu)
        print(message)

        if bool(hyp['learning_decay']) == True and epoch >= 1:
            if epoch_test_accu_list[-1] - epoch_test_accu_list[-2] < 0.001:
                learning_rate *= hyp['decay_factor']
            message = 'learning rate: %.8f' % learning_rate
            print(message)
    # CODE ENDS HERE
    #######################################
    return epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list


def logvals(plt_name, avg_train_loss_list, avg_train_accu_list, test_loss_list, test_accu_list):
    with open("./" + plt_name + "global_train_loss.txt", "w") as fp:
        for x in avg_train_loss_list:
            fp.write(str(x) + "\n")
    with open("./" + plt_name + "global_train_accu.txt", "w") as fp:
        for x in avg_train_accu_list:
            fp.write(str(x) + "\n")
    with open("./" + plt_name + "test_loss.txt", "w") as fp:
        for x in test_loss_list:
            fp.write(str(x) + "\n")
    with open("./" + plt_name + "test_accu.txt", "w") as fp:
        for x in test_accu_list:
            fp.write(str(x) + "\n")


# Function to be called at the start of the program
def main():

    # Name given to the logged text files
    plt_name = "logs/num_epoch_" + str(num_epochs) + "_batch_size_" + str(batch_size) + "_num_hidden_" + str(num_hidden) + "_lr_" + str(lr) + "_num_nodes_" + str(num_workers)

    # Uniformly distribute the data to all the clients
    # Each train_datasets is a list storing the dataset for each node similar to the centralized case
    train_datasets, test_datasets = uniformSampling()

    # hyperparameters to be passed to training routine
    hyp = {'num_epochs': num_epochs, 'batch_size': batch_size, 'learning_decay': True, 'learning_rate': lr, 'decay_factor': decay_factor}

    # setting the random seed
    np.random.seed(1)

    # initialize the parameters (we only have a single set of parameters to be updated for the model!)
    params = initialize(num_features, num_classes, num_hidden)

    # train the model
    epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list = trainDistributed(params, hyp, train_datasets, test_datasets)

    # log the loss and accuracy
    logvals(plt_name, epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list)


if __name__ == "__main__":
    main()
