# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:39:15 2017

@author: Xi Yu
"""

import numpy as np
import matplotlib.pyplot as plt

import textwrap
import math

numDatas = 20
sigma_Data = math.sqrt(1)
mu_Data =1
sigma0 = math.sqrt(1)
mu0 =1

mu_MAP= np.zeros(numDatas)
sigma = np.zeros(numDatas)
mu_ML = np.zeros(numDatas)
truemu = np.zeros(numDatas)
#Data = np.zeros(numDatas)
Data = []
for flip in range(numDatas):
    Data.append(np.random.normal(mu_Data,sigma_Data,1)[0])
    mu_ML[flip] = sum(Data)/len(Data)
    mu_1 = (mu0*(sigma_Data*sigma_Data))/((sigma0*sigma0)*len(Data)+sigma_Data*sigma_Data)
    mu_2 = (mu_ML[flip]*sigma0*sigma0*len(Data))/(sigma0*sigma0*len(Data)+sigma_Data*sigma_Data)
    mu_MAP[flip] = mu_1+mu_2
    sigma[flip] = (sigma_Data*sigma_Data*sigma0*sigma0)/(sigma0*sigma0*len(Data)+sigma_Data*sigma_Data)
    
print('ML of the Gaussian mean:' + str(mu_ML))
print('MAP of the Gaussian mean:' + str(mu_MAP))





for i in range(numDatas):
    truemu[i] = mu_Data
M = np.arange(1,numDatas+1,1)
p1 = plt.plot(M,truemu,'r')
p2 = plt.plot(M,mu_ML,'g')
p3 = plt.plot(M,mu_MAP,'b')
plt.legend((p1[0],p2[0],p3[0]),('truemu', 'ML Probability','MAP Probability',), fontsize=10)
plt.ylabel('mu',fontsize=15) 
plt.xlabel('the number of Data', fontsize=15)
plt.show()




    
    