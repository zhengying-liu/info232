"""
Created on Sat Mar 11 08:04:23 2017
Last revised: Feb 2, 2019

@author: isabelleguyon

This is an example of program that reads data and has a few display methods.

Add more views of the data getting inspired by previous lessons:
    Histograms of single variables
    Data matrix heat map
    Correlation matric heat map

Add methods of exploratory data analysis and visualization:
    PCA or tSNE
    two-way hierachical clustering (combine with heat maps)

The same class could be used to visualize prediction results, by replacing X by
the predicted values (the end of the transformation chain):
    For regression, you can 
        plot Y as a function of X.
        plot the residual a function of X.
    For classification, you can 
        show the histograms of X for each Y value.
        show ROC curves.
    For both: provide a table of scores and error bars.
"""

# Add the sample code in the path
mypath = "../ingestion_program"
from sys import argv, path
from os.path import abspath
import os
path.append(abspath(mypath))

# Graphic routines
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] # Red, lime, blue
cm = LinearSegmentedColormap.from_list('rgb', colors, N=3)

# Data types
import pandas as pd
import numpy as np

import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	# Converter class
	import data_converter
	# Mother class
	import data_manager
	# Typical score
	from sklearn.metrics import accuracy_score 

class DataManager(data_manager.DataManager):
    '''This class reads and displays data. 
       With class inheritance, we do not need to redefine the constructor,
       unless we want to add or change some data members.
       '''
       
    def __init__(self, basename="", input_dir=""):
        ''' New contructor.'''
        super(DataManager, self).__init__(basename, input_dir)
        # We added new members:
        self.feat_name = self.loadName (os.path.join(self.input_dir, basename + '_feat.name'))
        self.label_name = self.loadName (os.path.join(self.input_dir, basename + '_label.name'))
        
    def loadName (self, filename, verbose=False):
        ''' Get the variable name'''
        if verbose:  print("========= Reading " + filename)
        name_list = []
        if os.path.isfile(filename):
        	name_list = data_converter.file_to_array (filename, verbose=False)
        else:
        	n=self.info['feat_num']
        	name_list = [self.info['feat_name']]*n
        name_list = np.array(name_list).ravel()
        return name_list
        
    def __str__(self):
        val = "DataManager : " + self.basename + "\ninfo:\n"
        for item in self.info:
        	val = val + "\t" + item + " = " + str(self.info[item]) + "\n"
        val = val + "data:\n"
        val = val + "\tX_train = array"  + str(self.data['X_train'].shape) + "\n"
        val = val + "\tY_train = array"  + str(self.data['Y_train'].shape) + "\n"
        val = val + "\tX_valid = array"  + str(self.data['X_valid'].shape) + "\n"
        val = val + "\tY_valid = array"  + str(self.data['Y_valid'].shape) + "\n"
        val = val + "\tX_test = array"  + str(self.data['X_test'].shape) + "\n"
        val = val + "\tY_test = array"  + str(self.data['Y_test'].shape) + "\n"
        val = val + "feat_type:\tarray" + str(self.feat_type.shape) + "\n"
        val = val + "feat_idx:\tarray" + str(self.feat_idx.shape) + "\n"
        # These 2 lines are new:
        val = val + "feat_name:\tarray" + str(self.feat_name.shape) + "\n"
        val = val + "label_name:\tarray" + str(self.label_name.shape) + "\n"
        return val
    
    def toDF(self, set_name):
        ''' Change a given data subset to a data Panda's frame.
            set_name is 'train', 'valid' or 'test'.'''
        DF = pd.DataFrame(self.data['X_'+set_name])
        # For training examples, we can add the target values as
        # a last column: this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        if set_name == 'train':
            Y = self.data['Y_train']
            DF = DF.assign(target=Y)   
            # We modified the constructor to add self.feat_name, so we can also:
            # 1) Add a header to the data frame
            DF.columns=np.append(self.feat_name, 'target')
            # 2) Replace the numeric categories by the class labels
            DF = DF.replace({'target': dict(zip(np.arange(len(self.label_name)), self.label_name))})
        return DF
        
    ##### HERE YOU CAN IMPLEMENT YOUR OWN METHODS #####
     
    def DataStats(self, set_name):
    	''' Display simple data statistics.'''
    	DF = self.toDF(set_name)
    	return 0 # Return something better
    	
    def DataHist(self, set_name):
        ''' Show histograms.'''
        DF = self.toDF(set_name)
        return 0 # Return something better
    
    def ShowScatter(self, set_name):
        ''' Show scatter plots.'''
        DF = self.toDF(set_name)
        if set_name == 'train':
        	return 0 # Return something better
        else:
        	return 0 # Return something better

    def ShowSomethingElse(self):
        ''' Surprise me.'''
        # For your project proposal, provide
        # a sketch with what you intend to do written in English (or French) is OK.
        pass
        
    ##### END OF YOUR OWN METHODS ######################
    

    def ClfScatter(self, clf, dim1=0, dim2=1, title=''):
        '''(self, clf, dim1=0, dim2=1, title='')
        Split the training data into 1/2 for training and 1/2 for testing.
        Display decision function and training or test examples.
        clf: a classifier with at least a fit and a predict method
        like a sckit-learn classifier.
        dim1 and dim2: chosen features.
        title: Figure title.
        Returns: Test accuracy.
        '''
        X = self.data['X_train']
        Y = self.data['Y_train']
        F = self.feat_name
        # Split the data
        ntr=round(X.shape[0]/2)
        nte=X.shape[0]-ntr
        Xtr = X[0:ntr, (dim1,dim2)]
        Ytr = Y[0:ntr]
        Xte = X[ntr+1:ntr+nte, (dim1,dim2)]
        Yte = Y[ntr+1:ntr+nte]
        # Fit model in chosen dimensions
        clf.fit(Xtr, Ytr)
        # Compute the training score
        Yhat_tr = clf.predict(Xtr) 
        training_accuracy = accuracy_score(Ytr, Yhat_tr)
        # Compute the test score
        Yhat_te = clf.predict(Xte)  
        test_accuracy = accuracy_score(Yte, Yhat_te)       
        # Define a mesh    
        x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
        y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
        h = 0.1 # step
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
        Xgene = np.c_[xx.ravel(), yy.ravel()]
        # Make your predictions on all mesh grid points (test points)
        Yhat = clf.predict(Xgene) 
        # Make contour plot for all points in mesh
        Yhat = Yhat.reshape(xx.shape)
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, Yhat, cmap=plt.cm.Paired)
        # Overlay scatter plot of training examples
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr, cmap=cm)   
        plt.title('{}: training accuracy = {:5.2f}'.format(title, training_accuracy))
        plt.xlabel(F[dim1])
        plt.ylabel(F[dim2])
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, Yhat, cmap=plt.cm.Paired)
        # Overlay scatter plot of test examples
        plt.scatter(Xte[:, 0], Xte[:, 1], c=Yte, cmap=cm)   
        plt.title('{}: test accuracy = {:5.2f}'.format(title, test_accuracy))
        plt.xlabel(F[dim1])
        plt.ylabel(F[dim2])
        plt.subplots_adjust(left  = 0, right = 1.5, bottom=0, top = 1, wspace=0.2)
        plt.show()
        return test_accuracy


if __name__=="__main__":
    # You can use this to run this file as a script and test the DataManager
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    
    basename = 'Iris'
    D = DataManager(basename, input_dir)
    print(D)
    
    D.DataStats('train')