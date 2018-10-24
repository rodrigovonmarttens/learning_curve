# learning_curve

Python scripts to compute and plotting the calibrated learning curves 
for a given data set. This release has two scripts to indeed compute 
the learning curves, and one script to plot the learning curves. The 
available scripts are the following:

	-> learning_curve.py: general script for computing learning curves 
	for any linear template function. In case of using this script one 
	must modify it setting correctly IN THE SCRIPT the fiducial value 
	of the free parameters, the number of free parameters, the number 
	of loops the learning curve procedure will be performed, the 
	training set size, the data file (x, y, yerr), the covariance 
	matrix (single column with NxN rows), the template function and the 
	analytical components of the fisher matrix. The comments in the 
	script indicate where this variables must be setted. 
	
	-> learning_curve_linear.py: script for computing learning curves 
	for some specifics template functions (polynomial, log, inerse or 
	square). In this case is not necessary to modify the script itself, 
	but to write an auxiliar txt file. This auxiliary file must contain 
	seven lines with thw following informations: 

		1. name of the data file (must be in the data directory)
		2. name of the covariance matrix file (must be in the data directory)
		3. template (plynomial, log, inverse or square)
		4. order of the template function
		5. size of the training set
		6. number of loops
		7. name of the output

	See the file example_ini.txt as an example.
	
	In order to compute the learning curve with this script and the 
	auxiliary file, run in terminal the following command
	"python learning_curve_linear.py < auxiliary_file.txt"
	
	-> plot.py: script for plotting the learning curves from the output 
	files obtained using the script "learning_curve.py". Note that if 
	you have used the script in parallel you must merge all the output 
	files produced using "cat lc_train__* > lc_train.dat".
	
In learning_curve.py and learning_curve_linear.py if you don't want to 
compute the calibrated learning curves you must to comment the lines 
with the comment "# delta" in the end. 

I acknowledge Valerio Marra for valuble contributions and useful 
discussions.

Rodrigo von Marttens - rodrigovonmarttens@gmail.com
Postdoc researcher
University of Geneva
