#!/usr/bin/env python
"""
Script to plot the learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

dirFile = os.path.dirname(os.path.join('.','plot.py'))
plt.style.use(os.path.join(dirFile, 'Fig.mplstyle'))
########################################################################
# input
loops = # number of loops 
processes = # number of process used in parallelization
file_train_lc = # name of the merged train output file 
file_val_lc = # name of the merged validation output file 
file_delta_lc = # name of the merged delta output file                  # delta
data_train_lc = np.genfromtxt(file_train_lc, dtype=np.float64)
data_val_lc = np.genfromtxt(file_val_lc, dtype=np.float64)
data_delta_lc = np.genfromtxt(file_val_lc, dtype=np.float64)            # delta
########################################################################
# mean and std
N = loops * processes
size = len(data_train_lc[0])
lc_train_mean = np.zeros(size)
lc_val_mean = np.zeros(size)
lc_delta_mean = np.zeros(size)                                          # delta
lc_train_std = np.zeros(size)
lc_val_std = np.zeros(size)
lc_delta_std = np.zeros(size)                                           # delta
for i in np.arange(0, size):
	lc_train_mean[i] = np.mean(data_train_lc[:, i])
	lc_train_std[i] = np.std(data_train_lc[:, i])
	lc_val_mean[i] = np.mean(data_val_lc[:, i])
	lc_val_std[i] = np.std(data_val_lc[:, i])
	lc_delta_mean[i] = np.mean(data_delta_lc[:, i])                     # delta
	lc_delta_std[i] = np.std(data_delta_lc[:, i])                       # delta
lc_train_std = lc_train_std / np.sqrt(N)
lc_val_std = lc_val_std / np.sqrt(N)
lc_delta_std = lc_delta_std / np.sqrt(N)                                #delta
########################################################################
# learning curves plot
order = # number of free parameters used to compute the learning curve
x_plot = np.arange(order, size + order - 1)
plt.xlim(order, size + order - 1)
plt.ylim(0, 1.2)
plt.xlabel(r'Training set size')
plt.errorbar(x_plot, lc_train_mean, yerr = lc_train_std, color = 'red', label=r'Training set)')
plt.errorbar(x_plot, lc_val_mean, yerr = lc_val_std, color = 'blue', label=r'Validation set)')
plt.errorbar(x_plot, lc_delta_mean, yerr = lc_delta_std, color = 'green', label=r'$\delta$Validation set)')
plt.plot(x_plot, x_plot / x_plot, color = 'black', linestyle = ':')

leg=plt.legend(loc='best',fancybox=True)
leg.get_frame().set_alpha(0.9)

#plt.show()
#plt.savefig('fig.pdf', dpi=288)
