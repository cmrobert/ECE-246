from __future__ import print_function
import os
import numpy as np
from six.moves import cPickle as pickle
from matplotlib.pyplot import imread
import platform
from mnist import MNIST

def loadM(filename):
    """
    Load csv file into X array of features and y array of labels.
    Parameters
    --------------------
    filename -- string, filename
    """
    # determine filename
    dir = os.path.dirname(__file__)
    f = os.path.join(dir,'MNIST', filename)
    # load data
    with open(f, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")
    # separate features and labels
    X = data[:,1:]
    y = data[:,0].astype("int64")
    return X,y 

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))
    
    
    
def load_mnist() :
    """
    Load csv file into X array of features and y array of labels.
    Parameters
    --------------------
    filename -- string, filename
    """
    # determine filename
    dir = os.path.dirname(__file__)
    f = os.path.join(dir,'MNIST')
    # load data
    mndata = MNIST(f)
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train,X_test,y_train,y_test  
    