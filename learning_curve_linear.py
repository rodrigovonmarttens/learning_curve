"""
Script to compute the (calibrated) learning curve for a given data set using specific linear templates.
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import sys

rank = MPI.COMM_WORLD.Get_rank()
########################################################################
# plot configurations
dirFile = os.path.dirname(os.path.join('.','learning_curve_linear.py'))
plt.style.use(os.path.join(dirFile, 'Fig.mplstyle'))
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
########################################################################
# import data
file_data = input('Enter the data file: ')
print("\n"+ file_data)
file_sys = input('Enter the systematic covariance matrix: ')
print("\n"+ file_sys)
base = input('Enter the base functions (linear, log or inverse): ')
print("\n"+ base)
alpha_max = input('Enter the order of the reconstruction: ')
print("\n"+ alpha_max)
train_size = input('Enter the size of the train set: ')
print("\n"+ train_size)
loops = input('Enter the number of loops: ')
print("\n"+ loops)
name = input('Enter the name of outputs: ')
print("\n"+ name)
alpha_max = int(alpha_max)
train_size = int(train_size)
loops = int(loops)

data = np.genfromtxt('./data/'+file_data, dtype=np.float64)
x = data[:, 0]
y = data[:, 1]
yerr = data[:, 2]
sigma2 = yerr**2.
cov_stat = np.diag(sigma2)

data_sys = np.genfromtxt('./data/'+file_sys, dtype=np.float64)
size = int(np.sqrt(len(data_sys))-1)
cov_sys = data_sys.reshape(size+1,size+1)
########################################################################
# template
def template(x_data, y_data, yerr_data, covmat_stat, covmat_sys, x_vec):
	covmat = covmat_stat + covmat_sys
	Icovmat = inv(covmat)
	fisher = np.zeros([alpha_max + 1, alpha_max + 1])
	if (base == "linear"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				fisher[a][b] = np.dot((x_data)**b, np.dot(Icovmat, (x_data)**a))
	elif (base == "log"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				fisher[a][b] = np.dot(np.log(x_data)**b, np.dot(Icovmat, np.log(x_data)**a))
	elif (base == "inverse"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				fisher[a][b] = np.dot((1. / x_data)**b, np.dot(Icovmat, (1. / x_data)**a))
	elif (base == "square"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				fisher[a][b] = np.dot((x_data)**(b / 2.), np.dot(Icovmat, (x_data)**(a / 2.)))
	else:
		sys.exit("Error message: please choose a valid base functions (linear, log or inverse.)")
	Ifisher = inv(fisher)
	ca = np.zeros([alpha_max+1])
	if (base == "linear"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				ca[a] += np.dot(Ifisher[a][b], np.dot(y_data, np.dot(Icovmat, ((x_data)**b))))
	elif (base == "log"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				ca[a] += np.dot(Ifisher[a][b], np.dot(y_data, np.dot(Icovmat, (np.log(x_data)**b))))
	elif (base == "inverse"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				ca[a] += np.dot(Ifisher[a][b], np.dot(y_data, np.dot(Icovmat, ((1. / x_data)**b))))
	elif (base == "square"):
		for a in range(0, alpha_max + 1):
			for b in range(0, alpha_max + 1):
				ca[a] += np.dot(Ifisher[a][b], np.dot(y_data, np.dot(Icovmat, ((x_data)**(b / 2.)))))
	t = np.zeros([len(x_vec)])
	if (base == "linear"):
		for a in range(0, alpha_max + 1):
			t += ca[a] * (x_vec)**a
	elif (base == "log"):
		for a in range(0, alpha_max + 1):
			t += ca[a] * np.log(x_vec)**a
	elif (base == "inverse"):
		for a in range(0, alpha_max + 1):
			t += ca[a] * (1. / x_vec)**a
	elif (base == "square"):
		for a in range(0, alpha_max + 1):
			t += ca[a] * (x_vec)**(a / 2.)
	return(fisher, Ifisher, ca, t)
########################################################################
# learnig curve
def learnig_curve(x_data, y_data, yerr_data, covmat_stat, covmat_sys):
    data_size = len(y_data)
    val_size = data_size - train_size
    covmat_data = covmat_stat + covmat_sys
    lc_train = np.zeros([loops, train_size - alpha_max])
    lc_val = np.zeros([loops, train_size - alpha_max])
    lc_valdelta = np.zeros([loops, train_size - alpha_max])
    lc_delta = np.zeros([loops, train_size - alpha_max])
    for n in np.arange(0, loops):
        train_index = np.random.choice(data_size, train_size, replace=False)
        val_index = np.setdiff1d(np.arange(0,len(y_data)), train_index)
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
        fisher_val = template(x_val, y_val, yerr_val, covmat_val, np.zeros([val_size, val_size]), np.array([1.]))[0]
        aux_train = np.zeros([train_size - alpha_max])
        aux_val = np.zeros([train_size - alpha_max])
        aux_valdelta = np.zeros([train_size - alpha_max])
        aux_delta = np.zeros([train_size - alpha_max])
        for i in np.arange(0, train_size - alpha_max):
            setx_train = x_train[: alpha_max + 1 + i]
            sety_train = y_train[: alpha_max + 1 + i]
            setyerr_train = yerr_train[: alpha_max + 1 + i]
            setcovmat_train = covmat_train[: alpha_max + 1 + i, : alpha_max + 1 + i]
            settrain_size = len(setx_train)
            Isetcovmat_train = inv(setcovmat_train)
            template_train = template(setx_train, sety_train, setyerr_train, setcovmat_train, np.zeros([settrain_size, settrain_size]), setx_train)
            t_train = template_train[3]
            Ifisher_train = template_train[1]
            delta = 0.
            for a in range(0, alpha_max + 1):
                for b in range(0, alpha_max + 1):
                    delta += Ifisher_train[a][b] * fisher_val[a][b]
            chi2_train = np.dot(sety_train - t_train, np.dot(Isetcovmat_train, sety_train - t_train))
            template_val = template(setx_train, sety_train, setyerr_train, setcovmat_train, np.zeros([settrain_size, settrain_size]), x_val)
            t_val = template_val[3]
            chi2_val = np.dot(y_val - t_val, np.dot(Icovmat_val, y_val - t_val))
            aux_train[i] = chi2_train
            aux_val[i] = chi2_val
            aux_valdelta[i] = chi2_val - delta
            aux_delta[i] = delta
        lc_train[n] = aux_train
        lc_val[n] = aux_val
        lc_valdelta[n] = aux_valdelta
        lc_delta[n] = aux_delta
    for k in np.arange(0, train_size - alpha_max):
        lc_train[:, k] = lc_train[:, k] / (k + 1.)
    lc_val = lc_val / val_size
    lc_valdelta = lc_valdelta / val_size
    lc_delta = lc_delta / val_size
    lc_train_mean = np.zeros([train_size - alpha_max])
    lc_train_std = np.zeros([train_size - alpha_max])
    lc_val_mean = np.zeros([train_size - alpha_max])
    lc_val_std = np.zeros([train_size - alpha_max])
    lc_valdelta_mean = np.zeros([train_size - alpha_max])
    lc_valdelta_std = np.zeros([train_size - alpha_max])
    for i in np.arange(0, train_size - alpha_max):
        lc_train_mean[i] = np.mean(lc_train[:, i])
        lc_val_mean[i] = np.mean(lc_val[:, i])
        lc_valdelta_mean[i] = np.mean(lc_valdelta[:, i])
        lc_train_std[i] = np.std(lc_train[:, i]) / np.sqrt(loops)
        lc_val_std[i] = np.std(lc_val[:, i]) / np.sqrt(loops)
        lc_valdelta_std[i] = np.std(lc_valdelta[:, i]) / np.sqrt(loops)
    lc_train_last = lc_train[:, -1]
    lc_val_last = lc_val[:, -1]
    lc_valdelta_last = lc_valdelta[:, -1]
    lc_delta_last = lc_delta[:, -1]
    return(lc_train_mean, lc_train_std, lc_val_mean, lc_val_std, lc_valdelta_mean, lc_valdelta_std)
########################################################################
# learning curves plot
x_plot = np.arange(alpha_max + 1, train_size + 1, 1)
y_plot = learnig_curve(x, y, yerr, cov_stat, cov_sys)
ytrain_plot = y_plot[0]
yerrtrain_plot = y_plot[1]
yval_plot = y_plot[2]
yerrval_plot = y_plot[3]
yvaldelta_plot = y_plot[4]
yerrvaldelta_plot = y_plot[5]
plt.xlabel(r'Training set size')
plt.xlim(alpha_max + 1, train_size)
plt.ylim(0, 12)
plot_train = plt.errorbar(x_plot, ytrain_plot, yerr = yerrtrain_plot, color = 'red', label=r"$\chi^{2}_{\nu}$ (Training set)")
plot_val = plt.errorbar(x_plot, yval_plot, yerr = yerrval_plot, color = 'blue', label=r"$\tilde{\chi}^{2}_{\nu}$ (Validation set)")
plot_valdelta = plt.errorbar(x_plot, yvaldelta_plot, yerr = yerrvaldelta_plot, color = 'green', label=r"$\tilde{\chi}^{2}_{\delta}$ (Validation set)")
plot_one = plt.plot(np.arange(0., train_size, 1e-3), np.arange(0., train_size, 1e-3) / np.arange(0., train_size, 1e-3), color = 'black', linestyle = ':')

leg=plt.legend(loc='best',fancybox=True)
leg.get_frame().set_alpha(0.9)

#plt.savefig('./learningcurve_'+name+'_'+base+'_order'+str(alpha_max)+'_train'+str(train_size)+'_loops'+str(loops)+'.pdf', dpi=288)
plt.show()
