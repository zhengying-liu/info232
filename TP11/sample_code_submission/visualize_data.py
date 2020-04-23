# Example of visualization functions
# TP 11, info 232
# Isabelle Guyon
# 17 Avril 2020

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sys import path; path.append('../ingestion_program');
from data_io import read_as_df # found in ingestion_program directory
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]

def run_visualization(data_dir, data_name):
	''' Show a bunch of graphs.'''
	# Read data
	print('Read data')
	data = read_as_df(data_dir  + '/' + data_name)
	# Standartize data and change target value to numeric
	print('Standardize data')
	data_num = standardize_df(data)
	# Make a heat map
	print('Make hear map')
	sns.heatmap(data_num)
	plt.show()
	# Make scatter plots (limit to 5 variables if large number of variables)
	print('Make scatter plots')
	var_num = data.shape[1]-1
	if var_num>5: var_num = 5
	chosen_columns = list(data.columns[0:var_num])
	sns.pairplot(data, vars = chosen_columns, diag_kind="hist", hue="target")
	plt.show()
	print('Show correlation matrix')
	# Correlation matrix
	corr_mat = data_num[chosen_columns+['target']].corr(method='pearson')
	sns.heatmap(corr_mat, annot=True, center=0)
	plt.show()

def standardize_df(data):
	'''Make a standardized data frame with numerical target values.'''
	data_num = data.copy()  
	data_num['target'] = data_num['target'].astype('category')
	data_num['target'] = data_num['target'].cat.codes
	# Standardize; avoid standardizing the target values:
	target_values = data_num['target']
	data_num = (data_num-data_num.mean())/data_num.std()
	data_num['target'] = target_values
	return data_num

if __name__=="__main__":
	data_dir = '../iris'
	basename = 'iris'
	run_visualization(data_dir, basename) 
    
    
    
    