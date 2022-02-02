import numpy as np
import mnist
import matplotlib
import matplotlib.pyplot as plt
import copy
import random

# Load MNIST data
mndata = mnist.MNIST('data')
X_train, Y_train = mndata.load_training()
X_test, Y_test = mndata.load_testing() 

# Normalize to values between 0 and 1
X_train = np.asarray(X_train)/255
X_test = np.asarray(X_test)/255

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)


# some statistics about the dataset
num_train = np.shape(X_train)[0]
num_features = np.shape(X_train)[1]
num_test = np.shape(X_test)[0]
num_classes = len(set(Y_train))



##------------- Global parameters-----------------------
num_hidden = 20 # number of nodes in hidden layer 
num_epochs = 10 # number of epochs
batch_size = 10 #
lr = 0.2 # learning rate 
decay_factor = 0.99 # decay coefficient for learning rate
##------------------------------------------------------


def initialize(num_inputs,num_classes,num_hidden):
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



def prepare_datasets():
    s_train = np.arange(num_train)
    s_test = np.arange(num_test)
    np.random.shuffle(s_train)
    np.random.shuffle(s_test)
    X_train_s = np.array(X_train)[s_train.astype(int)]  #shuffled training dataset features
    Y_train_s = np.array(Y_train)[s_train.astype(int)]  #shuffled training dataset labels
    X_test_s = np.array(X_test)[s_test.astype(int)]     #shuffled test dataset features
    Y_test_s = np.array(Y_test)[s_test.astype(int)]     #shuffled test dataset labels
    train_dataset = [X_train_s[:,:],Y_train_s[:]]
    test_dataset = [X_test_s[:,:],Y_test_s[:]]
    return train_dataset, test_dataset


def ReLU(z):
    """
    ReLU activation function.
    inputs: z
    outputs: max(z,0)
    """
    r = z.clip(min=0)
    return r

def oneHotEncode(Y):
    E = np.zeros((num_classes,np.shape(np.asarray(Y))[0]))
    for i in range(np.shape(E)[1]):
        E[Y[i]][i] = 1
    return E

# Compute the cross entropy loss given estimate (Y_hat) and true labels (Y)
def compute_loss(Y, Y_hat):
    """
    Compute cross entropy loss
    """
    E = oneHotEncode(Y)
    L_sum = np.sum(np.multiply(E, np.log(Y_hat)))
    m = Y.shape[0]
    L = (-1*L_sum)/m 
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

    # Z1 = W1.dot(x) + b1
    cache["Z1"] = np.matmul(params["W1"], X.transpose()) + params["b1"]

    # # A1 = ReLU(Z1)
    cache["A1"] = ReLU(cache["Z1"])  

    # Z2 = W2.dot(A1) + b2
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]

    # A2 = softmax(Z2)
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache


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
    dA1_copy[cache["Z1"]<0] = 0 
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
    result = np.argmax(np.array(cache["A2"]).T,axis=1)
    accuracy = sum(result == y_data)/float(len(y_data))
    return loss, accuracy


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

# Centralized training routine
def trainCentralized(params, hyp, train_dataset,test_dataset):
    num_epochs = hyp['num_epochs']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list = [],[],[],[]
    for epoch in range(num_epochs):
        rand_indices = np.random.choice(num_train,num_train,replace=False)
        num_batch = int(num_train/batch_size)
        if bool(hyp['learning_decay']) == True:
            try:
                if epoch_test_accu_list[-1] - epoch_test_accu_list[-2] < 0.001:
                    learning_rate *= hyp['decay_factor']
            except:
                pass
            message = 'learning rate: %.8f' % learning_rate
            print(message)
        for batch in range(num_batch):
                index = rand_indices[batch_size*batch:batch_size*(batch+1)]
                x_batch = train_dataset[0][index]
                y_batch = train_dataset[1][index]
                params_grads = mini_batch_gradient(params, x_batch, y_batch)

                ## Perform update for the parameters (taking SGD step)
                params["W1"] = params["W1"] - learning_rate * params_grads["dW1"]
                params["b1"] = params["b1"] - learning_rate * params_grads["db1"]
                params["W2"] = params["W2"] - learning_rate * params_grads["dW2"]
                params["b2"] = params["b2"] - learning_rate * params_grads["db2"]

        train_loss, train_accu = eval(params,train_dataset[0],train_dataset[1])
        test_loss, test_accu = eval(params,test_dataset[0],test_dataset[1])                              
        epoch_train_loss_list.append(train_loss)
        epoch_train_accu_list.append(train_accu)                
        epoch_test_loss_list.append(test_loss)
        epoch_test_accu_list.append(test_accu)
        message = 'Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch+1, train_loss, train_accu, test_loss, test_accu)
        print(message)
    return epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list


def logvals(plt_name,avg_train_loss_list, avg_train_accu_list, test_loss_list, test_accu_list):
    '''
    Function to log the train and test performance of algorithm
    Output: Text files created in the directory containing logs

    PS: EXISTING FILES WITH THE SAME NAME WILL BE OVERWRITTEN!
    '''
    with open("./"+plt_name+"train_loss.txt","w") as fp:
        for x in avg_train_loss_list:
            fp.write(str(x)+"\n")
    with open("./"+plt_name+"train_accu.txt","w") as fp:
        for x in avg_train_accu_list:
            fp.write(str(x)+"\n")
    with open("./"+plt_name+"test_loss.txt","w") as fp:
        for x in test_loss_list:
            fp.write(str(x)+"\n")
    with open("./"+plt_name+"test_accu.txt","w") as fp:
        for x in test_accu_list:
            fp.write(str(x)+"\n")  


def main(): 

    # plt_name = 'fed_noniid_sgd_'+str(num_workers)+"_"+str(num_test_workers)+"_"+str(num_hidden)+"_"+str(num_iter)+"_relu_"
    plt_name = "num_epoch_"+str(num_epochs)+"_batch_size_"+str(batch_size)+"_num_hidden_"+str(num_hidden)+"_lr_"+str(lr)

    # prepare the training and test datasets (in format [features,labels])
    train_dataset,test_dataset = prepare_datasets()

    # hyperparameters to be passed to training routine
    hyp = {'num_epochs':num_epochs, 'batch_size':batch_size, 'learning_decay':True, 'learning_rate':lr, 'decay_factor':decay_factor}

    # setting the random seed
    np.random.seed(1)

    # initialize the parameters
    params = initialize(num_features,num_classes,num_hidden)

    # train the model
    epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list = trainCentralized(params, hyp, train_dataset, test_dataset)

    # log the loss and accuracy
    logvals(plt_name,epoch_train_loss_list, epoch_train_accu_list, epoch_test_loss_list, epoch_test_accu_list)

if __name__ == "__main__":
    main()        