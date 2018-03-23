# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:55:10 2017

@author: Xi Yu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
from scipy.stats import multivariate_normal


Data = np.loadtxt('GMDataSet_HW7.txt')
NumComponent = 3
NumData = 1000
feature = 5
Maxiteration = 200
Diffthresh = 1e-5
iff = 1

Ps = np.zeros(NumComponent)
means = np.zeros([NumComponent,feature])
sigma = np.zeros(NumComponent)



#Initalize the parameter
for i in range(NumComponent):
    means[i,:] = Data[np.random.randint(0,1000),:]
    Ps[i] = 1/NumComponent
    sigma[i]=1

pdf0 = np.zeros(1000)
pdf1 = np.zeros(1000)
pdf2 = np.zeros(1000)
for i in range(1000):
    pdf0[i] = multivariate_normal(means[0,:],sigma[0]*np.eye(feature)).pdf(Data[i,:])
    pdf1[i] = multivariate_normal(means[1,:],sigma[1]*np.eye(feature)).pdf(Data[i,:])
    pdf2[i] = multivariate_normal(means[2,:],sigma[2]*np.eye(feature)).pdf(Data[i,:])

Cik = np.zeros([NumData,NumComponent])



for j in range(NumData):
    Cik[j,0] = Ps[0]*pdf0[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])
    Cik[j,1] = Ps[1]*pdf1[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])
    Cik[j,2] = Ps[2]*pdf2[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])

Numberiteration = 0 
Diff = iff
#Diff>Diffthresh and
while( Numberiteration<=Maxiteration):
    
    meansold = means
    sigmaold = sigma
    Psold = Ps
    
    
    j0 = 0
    j1 = 0
    j2 = 0
    for i in range(1000):
        j0 = j0 + Cik[i,0]*Data[i,:]
        j1 = j1 + Cik[i,1]*Data[i,:]
        j2 = j2 + Cik[i,2]*Data[i,:]
    means[0,:] = j0/sum(Cik[:,0])
    means[1,:] = j1/sum(Cik[:,1])
    means[2,:] = j2/sum(Cik[:,2])
    
    i0 = 0
    i1 = 0
    i2 = 0
    for i in range(1000):
        xdiff0 = np.matrix(Data[i,:]-means[0,:])
        c0 = xdiff0@xdiff0.T
        i0 = i0 + Cik[i,0]*c0
        xdiff1 = np.matrix(Data[i,:]-means[1,:])
        c1 = xdiff1@xdiff1.T
        i1 = i1 + Cik[i,1]*c1
        xdiff2 = np.matrix(Data[i,:]-means[2,:])
        c2 = xdiff2@xdiff2.T
        i2 = i2 + Cik[i,2]*c2
    sigma[0] = i0/sum(Cik[:,0])
    sigma[1] = i1/sum(Cik[:,1])
    sigma[2] = i2/sum(Cik[:,2])
    
    Ps[0] = sum(Cik[:,0])/(sum(Cik[:,0])+sum(Cik[:,1])+sum(Cik[:,2]))
    Ps[1] = sum(Cik[:,1])/(sum(Cik[:,0])+sum(Cik[:,1])+sum(Cik[:,2]))
    Ps[2] = sum(Cik[:,2])/(sum(Cik[:,0])+sum(Cik[:,1])+sum(Cik[:,2]))

    
    for i in range(1000):
        pdf0[i] = multivariate_normal(means[0,:],sigma[0]*np.eye(feature)).pdf(Data[i,:])
        pdf1[i] = multivariate_normal(means[1,:],sigma[1]*np.eye(feature)).pdf(Data[i,:])
        pdf2[i] = multivariate_normal(means[2,:],sigma[2]*np.eye(feature)).pdf(Data[i,:])
        #Cik = np.zeros([NumData,NumComponent])



    for j in range(NumData):
        Cik[j,0] = Ps[0]*pdf0[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])
        Cik[j,1] = Ps[1]*pdf1[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])
        Cik[j,2] = Ps[2]*pdf2[j]/(Ps[0]*pdf0[j]+Ps[1]*pdf1[j]+Ps[2]*pdf2[j])
    
    Diff = sum(sum(abs(meansold-means))) + sum(abs(sigmaold-sigma)) + sum(abs(Psold-Ps))
      
    Numberiteration = Numberiteration + 1
    print(Numberiteration)
    print(Ps)
    print(means)
    #print(Diff)
    print(sigma)

def ratio(i,j):
    M = np.row_stack([means[i,:],means[j,:]])
    m = np.matrix(M)
    mean0 = np.mean(m,axis=0)
    b0 = m[0,:]- mean0
    Sb_0 = b0.T@b0
    b1 =  m[1,:]- mean0
    Sb_1 = b1.T@b1
    Sb = Ps[i]*Sb_0+Ps[j]*Sb_1
    Sw = sigma[i]*np.eye(feature)+sigma[j]*np.eye(feature)
    ratio = np.trace(Sb@np.linalg.inv(Sw))
    return ratio
    
ratio_01 = ratio(0,1)
ratio_02 = ratio(0,2)    
   
ratio_12 = ratio(1,2)



ratio_averge = (ratio_01+ratio_02+ratio_12)/3

fisher_ratio = np.array([1.253,1.603,1.061,0.821])
N = np.arange(2,6)
plt.ylabel('averge fisher ratio',fontsize=15) 
plt.xlabel('number of component', fontsize=15)
plt.plot(N,fisher_ratio,'b')
plt.plot(N,fisher_ratio,'bo')

