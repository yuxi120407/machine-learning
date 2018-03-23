# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:26:52 2017

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap

#Load Data
Data = np.loadtxt('HW6_Data.txt')

inputLayerSize = 2 
hiddenLayer1Size = 2
hiddenLayer2Size = 2
outputLayerSize = 1

W1 = np.random.randn(inputLayerSize+1,hiddenLayer1Size) #add a bais
W2 = np.random.randn(hiddenLayer1Size+1,hiddenLayer2Size)
W3 = np.random.randn(hiddenLayer2Size+1,outputLayerSize)
#W1 = np.random.uniform(size=(3, 2))
#W2 = np.random.uniform(size=(3, 1))

DataInput = Data[:,0:2]
DataOutput = Data[:,2]

Input = np.zeros([400,3])
arrayInput = np.zeros([400,3])

for i in range(400):
    Input[i,0] = 1

#add bias
arrayInput = np.matrix([Input[:,0],DataInput[:,0],DataInput[:,1]])
ArrayInput = arrayInput.T

#Sigmoid function
def sigmoid(x):
        return 1/(1+np.exp(-x))
    
#Gradient of sigmoid

def d_sig(x):
    return x*(1-x)

#back propagation
J = np.zeros(1000)
for i in range(1000):
    net1 = np.dot(ArrayInput,W1)
    output1 = sigmoid(net1)
    Output1 = np.array(output1)
    output1_bias = np.array([Input[:,0],Output1[:,0],Output1[:,1]]).T  #hiddenLayerSize = 2
    #output1_bias = np.array([Input[:,0],Output1[:,0],Output1[:,1],Output1[:,2]]).T #hiddenLayerSize = 3
    #output1_bias = np.array([Input[:,0],Output1[:,0],Output1[:,1],Output1[:,2],Output1[:,3]]).T #hiddenLayerSize = 4
    net2 = np.dot(output1_bias, W2) 
    output2 = sigmoid(net2)
    Output2 = np.array(output2)
    output2_bias = np.array([Input[:,0],Output2[:,0],Output2[:,1]]).T  #hiddenLayerSize = 2
    net3 = np.dot(output2_bias, W3)
    output3 = sigmoid(net3)
    
    dataOutput = np.matrix(DataOutput)
    error_output3 = np.array(dataOutput.T-output3)
    de_output3 = np.array(d_sig(output3))
    delta3 = error_output3*de_output3
    dJdW3 = output2_bias.T@delta3
    
    
    delta2 = delta3*W3[1:hiddenLayer2Size+1].T
    de_output2 = np.array(d_sig(Output2))
    local_error_2 = de_output2*delta2
    dJdW2 = output1_bias.T@local_error_2
    
    delta1 = delta3*W2[1:hiddenLayer1Size+1,1:hiddenLayer1Size+1].T
    de_output1 = np.array(d_sig(Output1))
    local_error_1 = de_output1*delta1
    dJdW1 = ArrayInput.T@local_error_1   
    
    W3 = W3 + 0.03*dJdW3
    W2 = W2 + 0.03*dJdW2
    W1 = W1 + 0.03*dJdW1
    J[i] = 0.5*(dataOutput.T-output3).T@(dataOutput.T-output3)/400
    
p1 = plt.scatter(Data[0:200,0],Data[0:200,1],color='blue')
p2 = plt.scatter(Data[200:399,0],Data[200:399,1],color='red')
plt.legend((p1, p2),
           ('species0', 'species1'),
           scatterpoints=1,
           loc='upper right',
           ncol=3,
           fontsize=8)

   
# plot the decision boundary
w1 = np.array(W1)
bias = -(w1[0][0]/w1[2][0])
ratio = -(w1[1][0]/w1[2][0])

bias1 = -(w1[0][1]/w1[2][1])
ratio1 = -(w1[1][1]/w1[2][1])
x1 = np.arange(-2, 2, 0.01)    
x2 = np.arange(-2, 2, 0.01)

X1 = np.arange(-2, 2, 0.01)    
X2 = np.arange(-2, 2, 0.01)

x2 = ratio*x1 + bias  
X2 = ratio1*X1 + bias1

plt.plot(x1,x2,color='blue')
plt.plot(X1,X2,color='red')



n = np.arange(0,1000,1)
fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(*[2,1,1])

ax.plot(n,J)