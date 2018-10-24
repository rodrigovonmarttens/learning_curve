# learning_curve

Python scripts to compute and plotting the calibrated learning curves 
for a given data set. This release has two scripts to indeed compute 
the learning curves, and one script to plot the learning curves. The 
available scripts are the following:

	1. learning_curve.py: general script for computing learning curves 
	for any linear template function. In case of using this script one 
	must modify it setting correctly IN THE SCRIPT the fiducial value 
	of the free parameters, the number of free parameters, the number 
	of loops the learning curve procedure will be performed, the 
	training set size, the data file (x, y, yerr), the covariance 
	matrix (single column with NxN rows), the template function and the 
	analytical components of the fisher matrix. 
	
	2. learning_curve_linear.py: script for computing learning curves 
	for some specifics template functions (polynomial, log or inerse). 
	In this case is not necessary to modify the script itself, but to 
	write an auxiliar txt file. 
	
	3. plot.py: script for plotting the learning curves from the output 
	files obtained using the previous scripts. Note that if you have 
	used the script in parallel you must merge all the output files 
	produced using "cat lc_train__* > lc_train.dat".
	
In learning_curve.py and learning_curve_linear.py if you don't want to 
compute the calibrated learning curves you must to comment the lines 
with the comment "# delta" in the end. 
