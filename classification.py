# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:19:44 2017

@author: Xi Yu
"""

#Import needed python libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
import time
from sklearn.metrics import confusion_matrix

# at the beginning:
start_time = time.time()


#Load Data
Data = np.loadtxt('HW5_dataset.txt')
training_Data = Data[0:140,0:7]
test_Data = Data[140:200,0:7]

training_s0 = np.zeros(7)
training_s0 = np.vstack((training_s0, training_Data[training_Data[:,0] == 0]))        
training_species0 = training_s0[1:73,1:7]     

training_s1 = np.zeros(7)
training_s1 = np.vstack((training_s1, training_Data[training_Data[:,0] == 1]))        
training_species1 = training_s1[1:70,1:7] 

testData = test_Data[:,1:7]

#computer mean and convariance
training_species0_mean = np.mean(training_species0, axis=0)
training_species0_cov = np.cov(training_species0.T)

#training_species0_cov = (training_species0-mean0).T@(training_species0-mean0)/72
training_species1_mean = np.mean(training_species1, axis=0)
training_species1_cov = np.cov(training_species1.T)


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
    alphabet = ['specise0','specise1']
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.title(title)
mu0 = np.matrix(training_species0_mean)
mu1 = np.matrix(training_species1_mean)
c1 = mu0@np.linalg.inv(training_species0_cov)@mu0.T
c2 = mu1@np.linalg.inv(training_species1_cov)@mu1.T
c = 0.25*(c1-c2)
cov = training_species0_cov+training_species1_cov
cov_inverse = np.linalg.inv(0.5*cov)
f = cov@cov_inverse
mu = mu1- mu0
w = cov_inverse@mu.T

g = np.zeros(140)
for i in range(140):
    g[i] = training_Data[i,1:7].T@w
    
trainingData_2D = np.zeros([140,2])
trainingData_2D[:,0] = training_Data[:,0] 
for i in range(140):
    trainingData_2D[i,1] = g[i]

training_s0 = np.zeros(2)
training_s0 = np.vstack((training_s0, trainingData_2D[trainingData_2D[:,0] == 0])) 
training_species0 = training_s0[1:73,:]   
#trainingData_2D = np.matrix([training_Data[:,0],trainingData_PCA_2D[:,0],trainingData_PCA_2D[:,1]])
training_s1 = np.zeros(2)
training_s1 = np.vstack((training_s1, trainingData_2D[trainingData_2D[:,0] == 1])) 
training_species1 = training_s1[1:69,:]
sample0 = np.arange(0,72)
sample1 = np.arange(0,68)
p1 = plt.scatter(sample0,training_species0[:,1],color='red')
p2 = plt.scatter(sample1,training_species1[:,1],color='blue')
plt.legend((p1, p2),
           ('species0', 'species1'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)


plt.show


predictData = np.zeros(200)
for i in range(200):
    if Data[i,1:7].T@w+15>=0:
        predictData[i] = 1
    if Data[i,1:7].T@w+15<0:
        predictData[i] = 0
        
trueData = np.zeros(200)        
for j in range(200):
    trueData[j] = Data[j,0]
d = np.zeros([2,2])
d = confusion_matrix(trueData,predictData)
plot_confusion_matrix(d,'confusion matrix for entire data sets',cmap=plt.cm.cool)
print("%f seconds" % (time.time() - start_time))
