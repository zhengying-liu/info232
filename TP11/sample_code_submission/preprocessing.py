"""
Created on Sat Mar 11 08:04:23 2017
Revised: Feb 2, 2019
Last revised: Apr 22, 2020

@author: isabelleguyon

This is an example of program that preprocesses data.
It does nothing it just copies the input to the output.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    re-combine features (PCA)
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

from sys import argv
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.base import BaseEstimator
    from zDataManager import DataManager # The class provided by binome 1
    # Note: if zDataManager is not ready, use the mother class DataManager
    from sklearn.decomposition import PCA

class Preprocessor(BaseEstimator):
    def __init__(self, standardize=True, transformer=None, num_feat=2):
    	'''Preprocessing that combines several ingredients:
		standardize=True/False: standartizes data or not (subtract mean and divide by stdev)
		transformer=None/'PCA'/'LDA': applies 'PCA', LDA or no transform 
		num_feat: select k best features. '''
    	pipe = []
    	if standardize:
    		pipe = [('standardization', StandardScaler())]
    	if transformer == 'PCA':
    		pipe = pipe + [('PCA', PCA(n_components=num_feat))]
    	elif transformer == 'LDA':
    		pipe = pipe + [('LDA', LinearDiscriminantAnalysis(n_components=num_feat))]
    	else:
    		pipe = pipe + [('kbest', SelectKBest(f_classif, k=num_feat))]
    	self.transformer = Pipeline(pipe)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X, y)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
    
if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../iris"
        output_dir = "../results"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'iris'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print(D)
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    D.feat_name = np.array(['Feat1', 'Feat2'])
    D.feat_type = np.array(['Numeric', 'Numeric'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print(D)
    
