# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:12:12 2017

@author: Xi Yu
"""

#Import needed python libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
from mpl_toolkits.mplot3d import Axes3D
import textwrap


#Load Data
EllipsoidsData = np.loadtxt('ellipsoids.txt')
SpheresData = np.loadtxt('spheres.txt')
SwissrollData = np.loadtxt('swissroll.txt')

#plot results
fig = plt.figure(figsize=(10,30))
ax = fig.add_subplot(*[3,1,1], projection= '3d')
ax.scatter(EllipsoidsData[:,0],EllipsoidsData[:,1],EllipsoidsData[:,2])
myTitle = 'Elliposids'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

ax = fig.add_subplot(*[3,1,2], projection= '3d')
ax.scatter(SpheresData[:,0],SpheresData[:,1],SpheresData[:,2])
myTitle = 'SpheresData'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

ax = fig.add_subplot(*[3,1,3], projection= '3d')
ax.scatter(SwissrollData[:,0],SwissrollData[:,1],SwissrollData[:,2])
myTitle = 'SwissrollData'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

#subtract mean
EllipsoidsData_x1 = EllipsoidsData[:,0]
EllipsoidsData_x2 = EllipsoidsData[:,1]
EllipsoidsData_x3 = EllipsoidsData[:,2]

SpheresData_x1 = SpheresData[:,0]
SpheresData_x2 = SpheresData[:,1]
SpheresData_x3 = SpheresData[:,2]

SwissrollData_x1 = SwissrollData[:,0]
SwissrollData_x2 = SwissrollData[:,1]
SwissrollData_x3 = SwissrollData[:,2]


EllipsoidsData_x1_mean = sum(EllipsoidsData_x1)/EllipsoidsData_x1.size
EllipsoidsData_x2_mean = sum(EllipsoidsData_x2)/EllipsoidsData_x2.size
EllipsoidsData_x3_mean = sum(EllipsoidsData_x3)/EllipsoidsData_x3.size

SpheresData_x1_mean = sum(SpheresData_x1)/SpheresData_x1.size
SpheresData_x2_mean = sum(SpheresData_x2)/SpheresData_x2.size
SpheresData_x3_mean = sum(SpheresData_x3)/SpheresData_x3.size

SwissrollData_x1_mean = sum(SwissrollData_x1)/SwissrollData_x1.size
SwissrollData_x2_mean = sum(SwissrollData_x2)/SwissrollData_x2.size
SwissrollData_x3_mean = sum(SwissrollData_x3)/SwissrollData_x3.size


EllipsoidsData_std_x1 = np.array([(EllipsoidsData_x1[m]-EllipsoidsData_x1_mean) for m in range(0,1500)])
EllipsoidsData_std_x2 = np.array([(EllipsoidsData_x2[m]-EllipsoidsData_x2_mean) for m in range(0,1500)])
EllipsoidsData_std_x3 = np.array([(EllipsoidsData_x3[m]-EllipsoidsData_x3_mean) for m in range(0,1500)])
EllipsoidsData_std_X = np.matrix([EllipsoidsData_std_x1,EllipsoidsData_std_x2,EllipsoidsData_std_x3])
EllipsoidsData_std = EllipsoidsData_std_X.T

SpheresData_std_x1 = np.array([(SpheresData_x1[m]-SpheresData_x1_mean) for m in range(0,1500)])
SpheresData_std_x2 = np.array([(SpheresData_x2[m]-SpheresData_x2_mean) for m in range(0,1500)])
SpheresData_std_x3 = np.array([(SpheresData_x3[m]-SpheresData_x3_mean) for m in range(0,1500)])
SpheresData_std_X = np.matrix([SpheresData_std_x1,SpheresData_std_x2,SpheresData_std_x3])
SpheresData_std = SpheresData_std_X.T

SwissrollData_std_x1 = np.array([(SwissrollData_x1[m]-SwissrollData_x1_mean) for m in range(0,1500)])
SwissrollData_std_x2 = np.array([(SwissrollData_x2[m]-SwissrollData_x2_mean) for m in range(0,1500)])
SwissrollData_std_x3 = np.array([(SwissrollData_x3[m]-SwissrollData_x3_mean) for m in range(0,1500)])
SwissrollData_std_X = np.matrix([SwissrollData_std_x1,SwissrollData_std_x2,SwissrollData_std_x3])
SwissrollData_std = SwissrollData_std_X.T


#compute covariance, eigenvals and eigenvecs
EllipsoidsData_covmat = np.cov(EllipsoidsData_std.T)
EllipsoidsData_eigen_vals, EllipsoidsData_eigen_vecs = np.linalg.eig(EllipsoidsData_covmat)

SpheresData_covmat = np.cov(SpheresData_std.T)
SpheresData_eigen_vals, SpheresData_eigen_vecs = np.linalg.eig(SpheresData_covmat)

SwissrollData_covmat = np.cov(SwissrollData_std.T)
SwissrollData_eigen_vals, SwissrollData_eigen_vecs = np.linalg.eig(SwissrollData_covmat)

#perform dimensionality reduction
EllipsoidsData_eigen_pairs = [(np.abs(EllipsoidsData_eigen_vals[i]), EllipsoidsData_eigen_vecs[:,i]) for i in range(len(EllipsoidsData_eigen_vals))]
EllipsoidsData_eigen_pairs.sort(reverse=True)
EllipsoidsData_w_2D = np.hstack((EllipsoidsData_eigen_pairs[0][1][:, np.newaxis], EllipsoidsData_eigen_pairs[1][1][:, np.newaxis]))
EllipsoidsData_PCA_2D = EllipsoidsData_std.dot(EllipsoidsData_w_2D)
EllipsoidsData_w_1D = np.hstack(EllipsoidsData_eigen_pairs[0][1][:, np.newaxis])
EllipsoidsData_PCA_1D = EllipsoidsData_std.dot(EllipsoidsData_w_1D)

SpheresData_eigen_pairs = [(np.abs(SpheresData_eigen_vals[i]), SpheresData_eigen_vecs[:,i]) for i in range(len(SpheresData_eigen_vals))]
SpheresData_eigen_pairs.sort(reverse=True)
SpheresData_w_2D = np.hstack((SpheresData_eigen_pairs[0][1][:, np.newaxis], SpheresData_eigen_pairs[1][1][:, np.newaxis]))
SpheresData_PCA_2D = SpheresData_std.dot(SpheresData_w_2D)
SpheresData_w_1D = np.hstack(SpheresData_eigen_pairs[0][1][:, np.newaxis])
SpheresData_PCA_1D = SpheresData_std.dot(SpheresData_w_1D)

SwissrollData_eigen_pairs = [(np.abs(SwissrollData_eigen_vals[i]), SwissrollData_eigen_vecs[:,i]) for i in range(len(SwissrollData_eigen_vals))]
SwissrollData_eigen_pairs.sort(reverse=True)
SwissrollData_w_2D = np.hstack((SwissrollData_eigen_pairs[0][1][:, np.newaxis], SwissrollData_eigen_pairs[1][1][:, np.newaxis]))
SwissrollData_PCA_2D = SwissrollData_std.dot(SwissrollData_w_2D)
SwissrollData_w_1D = np.hstack(SwissrollData_eigen_pairs[0][1][:, np.newaxis])
SwissrollData_PCA_1D = SwissrollData_std.dot(SwissrollData_w_1D)

#plot everything
fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(*[2,1,1])
ax.scatter([EllipsoidsData_PCA_2D[:,0]],[EllipsoidsData_PCA_2D[:,1]])
myTitle = 'Ellipsoids_2D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
ax = fig.add_subplot(*[2,1,2])
ax.scatter([EllipsoidsData_PCA_1D[0,:]], [np.zeros((1500,1))])
myTitle = 'Ellipsoids_1D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(*[2,1,1])
ax.scatter([SpheresData_PCA_2D[:,0]],[SpheresData_PCA_2D[:,1]])
myTitle = 'SpheresData_2D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
ax = fig.add_subplot(*[2,1,2])
ax.scatter([SpheresData_PCA_1D[0,:]], [np.zeros((1500,1))])
myTitle = 'SpheresData_1D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

fig = plt.figure(figsize=(5,10))
ax = fig.add_subplot(*[2,1,1])
ax.scatter([SwissrollData_PCA_2D[:,0]],[SwissrollData_PCA_2D[:,1]])
myTitle = 'SwissrollData_2D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
ax = fig.add_subplot(*[2,1,2])
ax.scatter([SwissrollData_PCA_1D[0,:]], [np.zeros((1500,1))])
myTitle = 'SwissrollData_1D'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
plt.show()

EllipsoidsData_eigen_tot  = sum(EllipsoidsData_eigen_vals)
EllipsoidsData_var_exp     = [(i/EllipsoidsData_eigen_tot) for i in sorted(EllipsoidsData_eigen_vals, reverse=True)]
Ellipsoids_cum_var_exp = np.cumsum(EllipsoidsData_var_exp)

plt.bar(range(1,4), EllipsoidsData_var_exp, color='rgb',alpha=0.5, align='center', label='individual explained variance')	
plt.step(range(1,4), Ellipsoids_cum_var_exp, alpha=0.5, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

SpheresData_eigen_tot = sum(SpheresData_eigen_vals)
SpheresData_var_exp     = [(i/SpheresData_eigen_tot) for i in sorted(SpheresData_eigen_vals, reverse=True)]
Spheres_cum_var_exp = np.cumsum(SpheresData_var_exp)

plt.bar(range(1,4), SpheresData_var_exp, color='rgb',alpha=0.5, align='center', label='individual explained variance')	
plt.step(range(1,4), Spheres_cum_var_exp, alpha=0.5, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

SwissrollData_eigen_tot = sum(SwissrollData_eigen_vals)
SwissrollData_var_exp     = [(i/SwissrollData_eigen_tot) for i in sorted(SwissrollData_eigen_vals, reverse=True)]
Swissroll_cum_var_exp = np.cumsum(SwissrollData_var_exp)

plt.bar(range(1,4), SwissrollData_var_exp, color='rgb',alpha=0.5, align='center', label='individual explained variance')	
plt.step(range(1,4), Swissroll_cum_var_exp, alpha=0.5, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()
