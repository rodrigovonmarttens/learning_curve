#!/usr/bin/env python
"""
Script to compute the (calibrated) learning curve for a given data set.
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
import sys
from scipy.optimize import minimize

rank = MPI.COMM_WORLD.Get_rank()
########################################################################
# Input parameters
a0_fid, b0_fid = # fiducial value for all free parameters
order = # number of free parameters

loops = # number of loops
train_size = # size of the training set

np.random.seed(rank)
########################################################################
# Input data
file_data = 'file_name.txt' # name of the data file with the appropriate extension
file_sys = 'covsys_name.txt' # name of the file with the systematic covariance matrix
data = np.genfromtxt(file_data, dtype=np.float64)
sys = np.genfromtxt(file_sys, dtype=np.float64)

x = data[:, 0]
y = data[:, 1]
yerr = data[:, 2]
sigma2 = yerr**2.
cov_stat = np.diag(sigma2)
order_sys = int(np.sqrt(len(sys)))
cov_sys = sys.reshape(order_sys,order_sys)
covmat = cov_stat + cov_sys
Icovmat = inv(covmat)
########################################################################
# Template function
def template(a0, b0, x_vec):
	t = a0 * xvec + b0 # template function
	return(t)

def log_likelihood(theta, x_data, y_data, yerr_data, icov_mat):
	a0_stat, b0_stat = theta
	model = template(a0_stat, b0_stat, x_data)
	return(-0.5 * np.dot(y_data - model, np.dot(icov_mat, y_data - model)))
########################################################################
# fisher matrix (necessary only if you want to compute the delta contribution)
def Fisher(x_data, cov_mat):
	Icov_mat = inv(cov_mat)
	fisher = fisher = np.zeros([order, order])
	fisher[0][0] = # insert the analytical expressions for the fisher matrix components
	fisher[0][1] = 
	fisher[1][0] = 
	fisher[1][1] = 
	return(fisher)
########################################################################
# learnig curve
def learnig_curve(x_data, y_data, yerr_data, covmat_stat, covmat_sys):
    data_size = len(y_data)
    val_size = data_size - train_size
    covmat_data = covmat_stat + covmat_sys
    lc_train = np.zeros([loops, train_size - (order - 1)])
    lc_val = np.zeros([loops, train_size - (order - 1)])
    lc_valdelta = np.zeros([loops, train_size - (order - 1)])
    lc_delta = np.zeros([loops, train_size - (order - 1)])
    for n in np.arange(0, loops):
        train_index = np.random.choice(data_size, train_size, replace=False)
        val_index = np.setdiff1d(np.arange(0, data_size), train_index)
        x_train = x_data[train_index]
        y_train = y_data[train_index]
        yerr_train = yerr_data[train_index]
        x_val = x_data[val_index]
        y_val = y_data[val_index]
        yerr_val = yerr_data[val_index]
        covmat_train = np.zeros([train_size, train_size])
        for i in np.arange(0, train_size):
            for j in np.arange(0, train_size):
                covmat_train[i][j] = covmat_data[train_index[i]][train_index[j]]
        Icovmat_train = inv(covmat_train)
        covmat_val = np.zeros([val_size, val_size])
        for i in np.arange(0, val_size):
            for j in np.arange(0, val_size):
                covmat_val[i][j] = covmat_data[val_index[i]][val_index[j]]
        Icovmat_val = inv(covmat_val)
        fisher_val = Fisher(x_val, covmat_val)                          #delta
        aux_train = np.zeros([train_size - (order - 1)])
        aux_val = np.zeros([train_size - (order - 1)])
        aux_delta = np.zeros([train_size - (order - 1)])
        for i in np.arange(0, train_size - (order - 1)):
            setx_train = x_train[: order + i]
            sety_train = y_train[: order + i]
            setyerr_train = yerr_train[: order + i]
            setcovmat_train = covmat_train[: order + i, : order + i]
            settrain_size = len(sety_train)
            Isetcovmat_train = inv(setcovmat_train)
            fisher_set_train = Fisher(setx_train, setcovmat_train)      # delta
            Ifisher_set_train = inv(fisher_set_train)                   # delta
            setnll_train = lambda *args: -log_likelihood(*args)
            setinitial_train = np.array([H0_fid, q0_fid, j0_fid, s0_fid]) + 0.1 * np.random.randn(order)
            setsoln_train = minimize(setnll_train, setinitial_train, args=(setx_train, sety_train, setyerr_train, Isetcovmat_train))
            H0_settrain, q0_settrain, j0_settrain, s0_settrain = setsoln_train.x
            print(H0_settrain, q0_settrain, j0_settrain)
            t_settrain = template(H0_settrain, q0_settrain, j0_settrain, s0_settrain, setx_train)
            delta = 0.                                                  # delta
            for a in range(0, order):                                   # delta
                for b in range(0, order):                               # delta
                    delta += Ifisher_set_train[a][b] * fisher_val[a][b] # delta
            chi2_train = np.dot(sety_train - t_settrain, np.dot(Isetcovmat_train, sety_train - t_settrain))
            t_val = template(H0_settrain, q0_settrain, j0_settrain, s0_settrain, x_val)
            chi2_val = np.dot(y_val - t_val, np.dot(Icovmat_val, y_val - t_val))
            aux_train[i] = chi2_train
            aux_val[i] = chi2_val
            aux_delta[i] = chi2_val - delta                             # delta
        lc_train[n] = aux_train
        lc_val[n] = aux_val
        lc_delta[n] = aux_delta                                         # delta
    for k in np.arange(0, train_size - (order - 1)):
        lc_train[:, k] = lc_train[:, k] / (k + 1.)
    lc_val = lc_val / val_size
    lc_delta = lc_delta / val_size                                      # delta
    print(lc_train, lc_val)
    return(lc_train, lc_val, lc_delta)                                  # delta
    #return(lc_train, lc_val) # in the case you are not computing the calibrated learning curve uncomment this line                                     
########################################################################
# output files
output = learnig_curve(x, y, yerr, cov_stat, cov_sys)
np.savetxt('./output/lc_train__'+str(rank)+'.dat', output[0])
np.savetxt('./output/lc_val__'+str(rank)+'.dat', output[1])
np.savetxt('./output/lc_delta__'+str(rank)+'.dat', output[2])           # delta

