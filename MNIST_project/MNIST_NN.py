# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:03:09 2017

@author: Xi Yu
"""
#%%
import numpy as np
import os
import struct
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
from array import array as pyarray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

Dataset_path = '/Users/Xi Yu/Desktop/machine learning_homework/project2/'
def load_mnist(dataset="training", digits=np.arange(10), path= Dataset_path, size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels, rows, cols

#Read the MNIST data.

#Download from: http://yann.lecun.com/exdb/mnist/.

#The format of the files is well documented in the website.

def readMnistData(filename, labels_filename, limit):
    imgs = open(filename, mode='rb')
    labels = open(labels_filename, mode='rb')
    imgs.read(4)
    labels.read(4)
    labels.read(4)
    img_num = int.from_bytes(imgs.read(4), byteorder="big")
    cols = int.from_bytes(imgs.read(4), byteorder="big")
    rows = int.from_bytes(imgs.read(4), byteorder="big")
    lists = [[],[]]
    if img_num > limit and limit != -1:
        img_num = limit
    #For each image convert it into a pixel_list and add it with the image's label to our "lists" list.
    for i in range(0,img_num):
        pixel_list = []
        for j in range(0, rows):
            for k in range(0, cols):
                pixel_list.append(int.from_bytes(imgs.read(1), byteorder="big"))
        label = int.from_bytes(labels.read(1), byteorder="big")
        lists[0].append(pixel_list)
        lists[1].append(label);
    return lists;

training_set_list = readMnistData("train-images.idx3-ubyte","train-labels.idx1-ubyte",6000)

training_images, training_labels,training_rows, training_cols  = load_mnist('training')
testing_images, testing_labels,testing_rows, testing_cols  = load_mnist('testing')

train_label = np.array(training_labels)
test_label = np.array(testing_labels)

n_images = len(training_images)
n_labels = len(training_labels)

test_n_images = len(testing_images)
test_n_labels = len(testing_labels)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
train_image = training_images.reshape((n_images, -1))
train_labels = train_label.reshape((n_labels, -1))
test_image = testing_images.reshape((test_n_images, -1))
test_labels = test_label.reshape((test_n_labels, -1))

features = train_image[:-1]
labels = train_labels[:-1]

training_image = train_image/255
testing_image = test_image/255

X_train = training_image[0:50000,:]
X_test = training_image[50000:60000,:]
y_train = train_labels[0:50000,:]
y_test = train_labels[50000:60000,:]

y_ture = np.zeros([10000,10])
for j, class_idx in enumerate(y_test):
    y_ture[j,int(class_idx)] = 1

#%%
# the parameters of size of each layers
inputlayersize = 784
hiddenLayerSize = 100
outputlayersize = 10
minibatch_size = 500
n_iter = 10
n_experiment = 10
eta = 1e-3 


#%%
def make_network():
    # Initialize weights with Standard Normal random variables
    #model = dict( W1=np.random.randn(inputlayersize+1, hiddenLayerSize),
                 #W2=np.random.randn(hiddenLayerSize+1, outputlayersize)
    model = dict( W1 = np.random.uniform(-1,1,(inputlayersize+1,hiddenLayerSize)),
                 W2 = np.random.uniform(-1,1,(hiddenLayerSize+1,outputlayersize))
    ) 

    return model

#def softmax(x):
    #return np.exp(x) / np.exp(x).sum()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ReLu(x):
        return np.maximum(x, 0)

def ReLuprime(x):
    return (x>0).astype(x.dtype)

def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]

def plot_confusion_matrix(conf_arr,title,cmap=plt.cm.cool):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    res = ax.imshow(np.array(norm_conf), cmap=cmap, interpolation='nearest')
    width, height = conf_arr.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')
    fig.colorbar(res)
    alphabet = ['0','1','2','3','4','5','6','7','8','9']
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.title(title)
    
def cross_entropy(prob,y_test,size):
    loss = np.zeros(size)

    for j, class_idx in enumerate(y_test):
        p = prob[j,class_idx]
        loss[j] = -np.log(p)
        
        J = np.sum(loss)/size
    
    return J



def forward(x, model):
    # Input to hidden
    x = np.append(1,x)
    h = x @ model['W1']
    # ReLU non-linearity
    #h[h < 0] = 0
    h = ReLu(h)
    h = np.append(1,h)

    # Hidden to output
    prob = softmax(h @ model['W2'])
    h = h[1:hiddenLayerSize+1]

    return h, prob


def backward(model, xs, hs, errs):
    """xs, hs, errs contain all informations (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    W2 = np.zeros([hiddenLayerSize+1, outputlayersize])
    hs = np.column_stack([np.ones(minibatch_size),hs])
    xs = np.column_stack([np.ones(minibatch_size),xs])
    dW2 = hs.T @ errs
    
    hs = hs[:,1:hiddenLayerSize+1]
    # Get gradient of hidden layer
    W2 = model['W2']
    
    dh = errs @ W2[1:hiddenLayerSize+1,:].T
    #dh[hs <= 0] = 0
    dh = dh * ReLuprime(hs)

    dW1 = xs.T @ dh

    return dict(W1=dW1, W2=dW2)

def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, class_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create probability distribution of true label
        y_true = np.zeros(outputlayersize)
        y_true[int(class_idx)] = 1.

        # Compute the gradient of softmax of the output layer
        error = y_true - y_pred

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(error)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))


def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    #model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-4
        model[layer] = model[layer] + eta * grad[layer]

    return model 


def sgd(model, X_train, y_train, minibatch_size):
    #for iter in range(n_iter):
        #print('SGD_Iteration {}'.format(iter))

        # Randomize data point
    X_train, y_train = shuffle(X_train, y_train)

    for i in range(0, X_train.shape[0], minibatch_size):
        # Get pair of (X, y) of the current minibatch/chunk
        X_train_mini = X_train[i:i + minibatch_size]
        y_train_mini = y_train[i:i + minibatch_size]

        model = sgd_step(model, X_train_mini, y_train_mini)

    return model

#%%
accs = np.zeros(n_experiment)
J = np.zeros(n_experiment)
model = make_network()
for k in range(n_experiment):
    # Reset model
    print('Iteration {}'.format(k))
    #model = make_network()

    # Train the model
    model = sgd(model, X_train, y_train, minibatch_size)

    y_pred = np.zeros_like(y_test)
    prob = np.zeros([y_test.size,10])
    
    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, prob[i,:] = forward(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob[i,:])
        y_pred[i] = y
        
    J[k] = cross_entropy(prob,y_test,y_test.size)
    #J[k] = log_loss(y_ture,prob,eps=1e-2)
    
    # Compare the predictions with the true labels and take the percentage
    accs[k] = (y_pred == y_test).sum() / y_test.size

print('Mean accuracy: {}, std: {}'.format(accs.mean(), accs.std()))
#%%
n = np.arange(n_experiment)
plt.plot(n,J)
plt.show()

m = np.arange(n_experiment)
plt.plot(m,accs)
plt.show()

p = np.zeros([10,10])
p = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(p,'confusion matrix for test datasets',cmap=plt.cm.cool)  




