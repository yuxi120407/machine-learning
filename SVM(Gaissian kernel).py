# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:04:17 2017

@author: Xi Yu
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm

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
    alphabet = ['0','1']
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.title(title)

X, y = make_moons(n_samples=5000, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#clf = svm.SVC(kernel='linear', C = 1.0)
clf = svm.SVC(kernel='rbf', C = 1.0,decision_function_shape='ovr',gamma = 1)
clf.fit(X_train,y_train)

#w = clf.coef_[0]
# = clf.intercept_[0]
#print(w)
#print(b)
p1 = plt.scatter(X[y==0,0], X[y==0, 1], color='red', marker = '^', alpha=0.5)
p2 = plt.scatter(X[y==1,0], X[y==1, 1], color='blue', marker = 'o', alpha=0.5)
plt.legend((p1, p2),
           ('species0', 'species1'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)

predict = np.zeros(1500)
distance = np.zeros(1500)
for i in range(1500):
    predict[i] = clf.predict([X_test[i,:]])
    distance[i] = clf.decision_function([X_test[i,:]])
account = 0
for i in range(1500):
    if predict[i] == y_test[i]:
        account = account+1
accuracy = account/1500

#xx = np.linspace(-1.5,2.5)
#yy = -w[0]/w[1]*xx - b/w[1]
#plt.plot(xx,yy)
#plt.show()

def plot_decision_boundary(pred_func):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

plot_decision_boundary(clf.predict)
w = clf.get_params(deep=True)