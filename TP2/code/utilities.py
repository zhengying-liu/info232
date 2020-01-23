# Load general libraries
import os, re
from glob import glob as ls
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import subprocess

def get_files(datadir, type = 'a'):
    ''' Function that gets all the files as a list. '''
    file_list = list()
    for root, dirs, files in os.walk(datadir):
        for name in files:
            if not re.search('.ipynb|.DS_Store', name) and re.search(type, name):
                file_list.append(os.path.join(root, name))
    return sorted(file_list, key = str)
    
def get_image(path):
    ''' Function that gets an image unsing PIL and sets white backgrounds to transparent. '''
    img = Image.open(path)
    data=img.getdata()  # Get a list of tuples
    newData=[]
    for a in data:
        a=a[:3] # Shorten to RGB
        if np.mean(np.array(a)) == 255: # the background is white
            a=a+(0,) # Put a transparent value in A channel (the fourth one)
        else:
            a=a+(255,) # Put a non- transparent value in A channel
        newData.append(a)
    img.putdata(newData) # Get new img ready
    return img

def show_images(all_files, filter = None, columns = 5, file_idx=None, show_num=False):
    ''' Function that shows the images whose names are given in the list all_files. 
        Optionally provide a filter, which is a function to apply to the images.
        The filter takes a PIL image as input and returns either a PIL image or a numpy array.
        The files are re-ordered according to file_idx.'''
    if file_idx:
    	all_files = all_files[file_idx]
    rows = math.ceil(1.*len(all_files)/columns)
    fig=plt.figure(figsize=(columns, rows))
    k=1
    for filename in all_files:
        img=get_image(filename)
        # Filter the image
        if filter:
            img = filter(img)
        fig.add_subplot(rows, columns, k)
        plt.imshow(img) 
        plt.tick_params(axis='both', labelsize=0, length = 0)
        plt.grid(b=False)
        if show_num:
            plt.xlabel(str(k),labelpad=-4)
        k=k+1
 
def preprocess_data(a_files, b_files, extract_features, standardize=False, verbose = False):      
    ''' Function that preprocesses all files and returns an array X with examples in lines
    	and features in column, and a column array Y with truth values.'''

    na = len(a_files)
    nb = len(b_files)
    # Probe dimensions
    img=get_image(a_files[1])
    f = extract_features(img, False)
    d = len(f)
    _X = np.zeros([na+nb, d])
    Y = np.zeros([na+nb, 1])
    
    k=0
    # Read and convert a_files
    for i in range(na):
    	if verbose: print(a_files[i])
    	img=get_image(a_files[i])
    	_X[k, :] = extract_features(img, verbose)
    	Y[k] = 1 # Apples are labeled 1
    	k=k+1
    	
    # Read and convert b_files
    for i in range(nb):
    	if verbose: print(b_files[i])
    	img=get_image(b_files[i])
    	_X[k, :] = extract_features(img, verbose)
    	Y[k] = -1 # Bananas are labeled -1
    	k=k+1
        
    # Eventually standardize
    if standardize:
    	if verbose: print('Standardize')
    	scaler = StandardScaler() 
    	X = scaler.fit_transform(_X)
    else:
    	X = _X
    return(X, Y)
    
def pretty_print(X, Y, column_names = None):
	''' Pretty print the data array X and the column target Y as a data table
		using column_names as header.'''
	XY = pd.DataFrame(np.append(X, Y, axis=1), columns=column_names)
	return XY
	
def heat_map(X, Y, column_names = None):
	''' Make a heat map of the data array X and the column target Y as a data table
		using column_names as header.'''
	XY = pd.DataFrame(np.append(X, Y, axis=1), columns=column_names)
	plt.figure(figsize=(5,10))
	sns.heatmap(XY, annot=True, fmt='f', cmap='RdYlGn')
	
def split_data(X, Y, verbose = True, seed=0):
	''' Make a 50/50 training/test data split (stratified).
		Return the indices of the split train_idx and test_idx.'''
	SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
	for train_index, test_index in SSS.split(X, Y):
		if verbose: print("TRAIN:", train_index, "TEST:", test_index)
		else: pass
	return (train_index, test_index)
	
def make_scatter_plot(X, F, train_index, test_index, filter=None, predicted_labels=[], show_diag=False, axis='normal'):
	'''This scatter plot function allows us to show the images.
		predicted_labels can either be: 
				- None (queries shown as question marks)
				- a vector of +-1 predicted values
				- the string "GroundTruth" (to display the test images).
		Other optional arguments: 
			show_diag: add diagonal dashed line if True.
			axis: make axes identical if 'square'.'''
	fruit = np.array(['B', 'A'])
	fig, ax = plt.subplots()
	# Plot training examples
	x = X[train_index,0]
	y = X[train_index,1]
	f = F[train_index]
	ax.scatter(x, y, s=750, marker='o') 

	for x0, y0, path in zip(x, y, f):
		img = get_image(path)
		if filter:
			img = filter(img)
		ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
		ax.add_artist(ab)
    
	# Plot test examples
	x = X[test_index,0]
	y = X[test_index,1]

	if len(predicted_labels)>0 and not(predicted_labels == "GroundTruth"):
		label = (predicted_labels+1)/2
		ax.scatter(x, y, s=250, marker='s', color='k') 
		for x0, y0, lbl in zip(x, y, label):
			ax.text(x0-0.05, y0-0.05, fruit[int(lbl)], color="w", fontsize=12, weight='bold')
	elif predicted_labels == "GroundTruth":
		f = F[test_index]
		ax.scatter(x, y, s=500, marker='s', color='k') 
		for x0, y0, path in zip(x, y, f):
			img = get_image(path)
			img = filter(img)
			ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
			ax.add_artist(ab)
	else: 	# Plot UNLABELED test examples
		f = F[test_index]
		ax.scatter(x, y, s=250, marker='s', c='k') 
		ax.scatter(x, y, s=100, marker='$?$', c='w') 
   	
	if axis == 'square':			
		ax.set_aspect('equal', adjustable='box')
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.xlabel('$x_1$ = Redness')
	plt.ylabel('$x_2$ = Elongation')
	
	# Add line on the diagonal
	if show_diag:
		plt.plot([-3, 3], [-3, 3], 'k--')
	return

def crop_image(img, crop_size = 32):
    ''' Crop a PIL image to crop_size x crop_size.'''
    # First determine the bounding box
    width, height = img.size
    new_width = 1.*crop_size
    new_height = 1.*crop_size
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    right = round((width + new_width)/2)
    bottom = round((height + new_height)/2)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def difference_filter(img):
    '''Extract a numpy array D = R-(G+B)/2 from a PIL image.'''
    img = crop_image(img)
    M = np.array(img)
    R = 1.*M[:,:,0]; G = 1.*M[:,:,1]; B = 1.*M[:,:,2]
    D = R-(G+B)/2
    return D

# Solution of question 4.2
def value_filter(img):
    '''Extract a numpy array V = (R+G+B)/3 from a PIL image.'''
    img = crop_image(img)
    M = np.array(img)
    R = 1.*M[:,:,0]; G = 1.*M[:,:,1]; B = 1.*M[:,:,2]
    V = (R+G+B)/3
    return V

def foreground_filter(img, theta = 2/3):
    '''Extract a numpy array with True as foreground 
    and False as background from a PIL image.
    Parameter theta is a relative binarization threshold.'''
    D = difference_filter(img)
    V = value_filter(img) 
    F0 = np.maximum(D, V)
    threshold = theta*(np.max(F0) - np.min(F0))
    F = F0>threshold
    return F

def transparent_background_filter(img, theta = 2/3):
    '''Create a cropped image with transparent background.'''
    F = foreground_filter(img)
    img = crop_image(img)
    data=img.getdata()  # Get a list of tuples
    newData=[]
    for a, f in zip(data, F.ravel()):
        a=a[:3] # Shorten to RGB
        if not f: # background pixel
            a=(255,255,255,0)
            #a=a+(0,) # Put a transparent value in A channel (the fourth one)
        else:
            a=a+(255,) # Put a non-transparent value in A channel
        newData.append(a)
    new_img = Image.new('RGBA', (32,32))
    new_img.putdata(newData) # Get new img ready
    return new_img

def get_redness(img):
    '''Extract the scalar value redness from a PIL image.'''
    D = difference_filter(img)
    F = foreground_filter(img)
    redness = np.mean(D[F])
    return redness

def get_colors(img):
    '''Extract various colors from a PIL image.'''
    F = foreground_filter(img)
    img = crop_image(img)
    M = np.array(img)
    R = 1.*M[:,:,0]
    G = 1.*M[:,:,1]
    B = 1.*M[:,:,2]
    Mx = np.maximum(np.maximum(R, G), B)
    Mn = np.minimum(np.minimum(R, G), B)
    C = Mx - Mn # Chroma
    D1 = R-(G+B)/2
    D2 = G - B
    D3 = G-(R+B)/2
    D4 = B - R   
    D5 = B-(G+R)/2
    D6 = R - G
    # Hue
    #H1 = np.divide(D2, C)
    #H2 = np.divide(D4, C)
    #H3 = np.divide(D6, C)        
    # Luminosity
    V = (R+G+B)/3
    # Saturation
    #S = np.divide(C, V)
    # We avoid divisions so we don't get divisions by 0
    # Now compute the color features
    r = np.mean(R[F])
    g = np.mean(G[F])
    b = np.mean(B[F])
    mx = np.mean(Mx[F])
    mn = np.mean(Mn[F])
    c = np.mean(C[F])
    d1 = np.mean(D1[F])
    d2 = np.mean(D2[F])
    d3 = np.mean(D3[F])
    d4 = np.mean(D4[F])
    d5 = np.mean(D5[F])
    d6 = np.mean(D6[F])
    h1 = d2/c
    h2 = d4/c
    h3 = d6/c  
    v = np.mean(V[F])
    s = c/v
    return np.array([r,g,b,mx,mn,c,d1,d2,d3,d4,d5,d6,h1,h2,h3,v,s])

def get_elongation(img):
    '''Extract the scalar value elongation from a PIL image.'''
    F = foreground_filter(img)
    # Find the array indices of the foreground image pixels
    xy = np.argwhere(F)
    # We first center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # We now apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    elongation = s[0]/s[1]
    return elongation

def get_shape(img):
    '''Extract shape parameters from a PIL image.'''
    F = foreground_filter(img)
    # Find the array indices of the foreground image pixels
    xy = np.argwhere(F)
    # We first center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # We now apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    elongation = s[0]/s[1]
    surface = s[0]*s[1]
    return np.array([s[0],s[1],elongation,surface])

def extract_features(img, verbose = True):
    '''Take a PIL image and return two features of the foreground: redness and elongation.'''
    redness = get_redness(img)
    if verbose: print('redness={0:5.2f}'.format(redness))
    elongation = get_elongation(img)
    if verbose: print('elongation={0:5.2f}'.format(elongation))
    return [redness, elongation]

def extract_more_features(img, verbose = True):
    '''Take a PIL image and return many features of the foreground: color and shape features.'''
    color_features = get_colors(img)
    shape_features = get_shape(img)
    return np.append(color_features,shape_features)

def extract_raw_image(img, verbose = True):
    '''Take a PIL image and return a vector of flattened pixel values.'''
    return np.array(img).ravel()

def extract_cropped_image(img, verbose = True):
    '''Take a PIL image and return a vector of flattened pixel values of a cropped image.'''
    img = crop_image(img)
    return np.array(img).ravel()

def error_rate(solution, prediction):
    '''Compute the error rate between two vectors.'''
    e = np.mean(solution!=prediction)
    return e

def simple_linear_predict(X):
    '''Function taking an array X of unlabeled examples as input and returning the predicted label vector y.'''
    G = X[:,0]-X[:,1]
    Y = np.ones([X.shape[0],1])
    Y[G<0] = -1
    return Y

def run_CV(X, Y, classification_method, training_method=None, n=10, verbose=False):
    '''Repeat several times a 2-way split of the data into training and test set.
       n is the number of repeats. Compute the mean and error bar.
       You may provide only a classification method making predictions
       (equivalent to classifier.predict in scikit-learn style) if your
       method does not involce any training. Otherwise, also supply a
       training method (equivalent to classifier.fit in scikit-learn style).'''
    SSS = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=5)
    E_tr = np.zeros([n,1])
    E_te = np.zeros([n,1])
    k = 0
    for train_index, test_index in SSS.split(X, Y):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        if training_method:
            training_method(Xtrain, Ytrain.ravel()) 
        Ytrain_predicted = classification_method(Xtrain)
        e_tr = error_rate(Ytrain.ravel(), Ytrain_predicted.ravel())
        if verbose:
            print("TRAINING ERROR RATE:", e_tr)
        E_tr[k] = e_tr
        Ytest_predicted = classification_method(Xtest)
        e_te = error_rate(Ytest.ravel(), Ytest_predicted.ravel())
        if verbose:
            print("TEST ERROR RATE:", e_te)
        E_te[k] = e_te
        k = k+1
    
    e_tr_ave = np.mean(E_tr)
    e_te_ave = np.mean(E_te)
    std_tr = np.std(E_tr)
    std_te = np.std(E_te)
    # Error bars
    n_tr = len(Ytrain)
    sigma_tr = np.maximum(std_tr, np.sqrt(e_tr_ave * (1-e_tr_ave) / n_tr))
    n_te = len(Ytest)
    sigma_te = np.maximum(std_te, np.sqrt(e_te_ave * (1-e_te_ave) / n_te))
    print("TRAINING ERROR RATE: {0:.2f} +- {1:.2f}".format(e_tr_ave, sigma_tr))
    print("TEST ERROR RATE: {0:.2f} +- {1:.2f}".format(e_te_ave, sigma_te))
    return (e_tr_ave, sigma_tr, e_te_ave, sigma_te)
    
def data_to_csv(X, Y, column_names = None, file_name=None):
	''' Save the data array X and the column target Y as a csv data table
		using column_names as header to file file_name.'''
	if not column_names: column_names = range(1+X.shape[1])
	df = pd.DataFrame(np.append(X, Y, axis=1), columns=column_names)
	df.to_csv(file_name, index=False)
	return df

def check_datasets(data_list):
    ''' Create a table with dataset statistics.'''
    col = ['Dataset', 'num. examples', 'num. features', 'num. apples', 'num. bananas']
    data = []
    for file in data_list:
        df = pd.read_csv(file)
        N, F = df.shape
        Na = sum(df.iloc[:,-1]==1)
        Nb = sum(df.iloc[:,-1]==-1)
        data.append([os.path.basename(file)[:-9],N,F,Na,Nb])
        stat_df = pd.DataFrame(data, columns = col)
    return stat_df

def df_cross_validate(df, sklearn_model, sklearn_metric, n=10, verbose=False):
    '''Repeat several times a 2-way split of the data into training and test set.
       n is the number of repeats. Compute the mean and error bar.
       Provide a sklearn model (sklearn = scikit-learn) and a sklearn performance metric.
       Note that a performance metric can either be an loss (error rate) or an accuracy (success rate).'''
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    SSS = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=5)
    Perf_tr = np.zeros([n,1])
    Perf_te = np.zeros([n,1])
    k = 0
    for train_index, test_index in SSS.split(X, Y):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        sklearn_model.fit(Xtrain, Ytrain.ravel()) 
        Ytrain_predicted = sklearn_model.predict(Xtrain)
        perf_tr = sklearn_metric(Ytrain.ravel(), Ytrain_predicted.ravel())
        if verbose:
            print("TRAINING PERFORMANCE METRIC", perf_tr)
        Perf_tr[k] = perf_tr
        Ytest_predicted = sklearn_model.predict(Xtest)
        perf_te = sklearn_metric(Ytest.ravel(), Ytest_predicted.ravel())
        if verbose:
            print("TEST PERFORMANCE METRIC:", perf_te)
        Perf_te[k] = perf_te
        k = k+1
    
    perf_tr_ave = np.mean(Perf_tr)
    perf_te_ave = np.mean(Perf_te)
    sigma_tr = np.std(Perf_tr)
    sigma_te = np.std(Perf_te)
    if verbose:
        metric_name = sklearn_metric.__name__.upper()
        print("*** AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(metric_name, perf_tr_ave, sigma_tr))
        print("*** AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(metric_name, perf_te_ave, sigma_te))
    return (perf_tr_ave, sigma_tr, perf_te_ave, sigma_te)
	
def systematic_data_experiment(data_name, all_data_df, sklearn_model, sklearn_metric):
    '''Run cross-validation on a bunch of datasets and collect the results.'''
    print(sklearn_model.__class__.__name__.upper())
    result_df = pd.DataFrame(columns =["perf_tr", "std_tr", "perf_te", "std_te"])
    for name, df in zip(data_name, all_data_df): 
        result_df.loc[name] = df_cross_validate(df, sklearn_model, sklearn_metric)
    return result_df

def systematic_model_experiment(data_df, model_name, model_list, sklearn_metric):
    '''Run cross-validation on a bunch of models and collect the results.'''
    result_df = pd.DataFrame(columns =["perf_tr", "std_tr", "perf_te", "std_te"])
    for name, model in zip(model_name, model_list): 
        result_df.loc[name] = df_cross_validate(data_df, model, sklearn_metric)
    return result_df

def analyze_model_experiments(result_df):
    tebad = result_df.perf_te < result_df.perf_te.median()
    trbad = result_df.perf_tr < result_df.perf_tr.median()
    overfitted = tebad & ~trbad
    underfitted = tebad & trbad
    result_df['Overfitted'] = overfitted
    result_df['Underfitted'] = underfitted
    return result_df.style.apply(highlight_above_median)

def highlight_above_median(s):
    '''Highlight values in a series above their median. '''
    medval = s.median()
    return ['background-color: yellow' if v>medval else '' for v in s]
    

#####
# Hierarchical clustering
import matplotlib as mpl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import string
import time
import sys, os
import getopt

def heatmap(X, row_method, column_method, row_metric, column_metric, color_gradient, default_window_hight = 70, default_window_width = 12):

    print(
        "\nPerforming hierarchical clustering using {} for columns and {} for rows".
        format(column_metric, row_metric))
    """
    This below code is based in large part on the protype methods:
    http://old.nabble.com/How-to-plot-heatmap-with-matplotlib--td32534593.html
    http://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre
    x is an m by n ndarray, m observations, n genes
    """

    ### Define variables
    x = np.array(X)
    column_header = column_header = [str(dataset) for dataset in list(X)]  # X.columns.values
    row_header = [str(model) for model in list(X.index)]  # X.index

    ### Define the color gradient to use based on the provided name
    n = len(x[0])
    m = len(x)
    if color_gradient == 'red_white_blue':
        cmap = plt.cm.bwr
    if color_gradient == 'red_black_sky':
        cmap = RedBlackSkyBlue()
    if color_gradient == 'red_black_blue':
        cmap = RedBlackBlue()
    if color_gradient == 'red_black_green':
        cmap = RedBlackGreen()
    if color_gradient == 'yellow_black_blue':
        cmap = YellowBlackBlue()
    if color_gradient == 'seismic':
        cmap = plt.cm.seismic
    if color_gradient == 'green_white_purple':
        cmap = plt.cm.PiYG_r
    if color_gradient == 'coolwarm':
        cmap = plt.cm.coolwarm

    ### Scale the max and min colors so that 0 is white/black
    vmin = x.min()
    vmax = x.max()
    vmax = max([vmax, abs(vmin)])
    # vmin = vmax*-1
    # norm = mpl.colors.Normalize(vmin/2, vmax/2) ### adjust the max and min to scale these colors
    norm = mpl.colors.Normalize(vmin, vmax)
    ### Scale the Matplotlib window size
    fig = plt.figure(figsize=(default_window_width, default_window_hight))  
    ### could use m,n to scale here
    color_bar_w = 0.015  ### Sufficient size to show

    ## calculate positions for all elements
    # axm, placement of heatmap for the data matrix
    [axm_x, axm_y, axm_w, axm_h] = [0.05, 0.95, 1, 1]
    width_between_axm_axr = 0.01
    text_margin = 0.1 # space between color bar and feature names
    
    # axr, placement of row side colorbar
    [axr_x, axr_y, axr_w, axr_h] = [0.31, 0.1, color_bar_w, 0.6]  
    ### second to last controls the width of the side color bar - 0.015 when showing
    axr_x =  axm_x + axm_w + width_between_axm_axr + text_margin
    axr_y = axm_y
    axr_h = axm_h
    width_between_axr_ax1 = 0.004
    
    # ax1, placement of dendrogram 1, on the right of the heatmap
    #if row_method != None: w1 =
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05, 0.22, 0.2, 0.6]
    ax1_x = axr_x + axr_w + width_between_axr_ax1
    ax1_y = axr_y
    ax1_h = axr_h
    ### The second value controls the position of the matrix relative to the bottom of the view
    width_between_ax1_axr = 0.004
    height_between_ax1_axc = 0.004  ### distance between the top color bar axis and the matrix

    # axc, placement of column side colorbar
    [axc_x, axc_y, axc_w, axc_h] = [0.4, 0.63, 0.5, color_bar_w]  
    ### last one controls the height of the top color bar - 0.015 when showing
    axc_x = axm_x
    axc_y = axm_y - axc_h - width_between_axm_axr - text_margin
    axc_w = axm_w
    height_between_axc_ax2 = 0.004

    # ax2, placement of dendrogram 2, on the top of the heatmap
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3, 0.72, 0.6, 0.15]
    ### last one controls height of the dendrogram
    ax2_x = axc_x
    ax2_y = axc_y - axc_h - ax2_h - height_between_axc_ax2
    ax2_w = axc_w

    # axcb - placement of the color legend
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.07, 0.88, 0.18, 0.09]
    axcb_x = ax1_x
    axcb_y = ax2_y
    axcb_w = ax1_w
    axcb_h = ax2_h

    # Compute and plot bottom dendrogram
    if column_method != None:
        start_time = time.time()
        d2 = dist.pdist(x.T)
        D2 = dist.squareform(d2)
        ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
        Y2 = sch.linkage(D2, method=column_method, metric=column_metric)  
        ### array-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z2 = sch.dendrogram(Y2, orientation='bottom')
        ind2 = sch.fcluster(Y2, 0.7 * max(Y2[:, 2]), 'distance')  
        ### This is the default behavior of dendrogram
        ax2.set_xticks([])  ### Hides ticks
        ax2.set_yticks([])
        time_diff = str(round(time.time() - start_time, 1))
        print('Column clustering completed in {} seconds'.format(time_diff))
    else:
        ind2 = ['NA'] * len(column_header)  
        ### Used for exporting the flat cluster data

    # Compute and plot right dendrogram.
    if row_method != None:
        start_time = time.time()
        d1 = dist.pdist(x)
        D1 = dist.squareform(d1)  # full matrix
        ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True)  
        # frame_on may be False
        Y1 = sch.linkage(D1, method=row_method, metric=row_metric)  
        ### gene-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z1 = sch.dendrogram(Y1, orientation='right')
        ind1 = sch.fcluster(Y1, 0.7 * max(Y1[:, 2]), 'distance')  
        ### This is the default behavior of dendrogram
        # print 'ind1', ind1
        ax1.set_xticks([])  ### Hides ticks
        ax1.set_yticks([])
        time_diff = str(round(time.time() - start_time, 1))
        print('Row clustering completed in {} seconds'.format(time_diff))
    else:
        ind1 = ['NA'] * len(row_header)  
        ### Used for exporting the flat cluster data

    # Plot distance matrix.
    axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])  
    # axes for the data matrix
    xt = x
    if column_method != None:
        idx2 = Z2['leaves']  
        ### apply the clustering for the array-dendrograms to the actual matrix data
        xt = xt[:, idx2]
        # print 'idx2', idx2, len(idx2)
        # print 'ind2', ind2, len(ind2)
        ind2 = [ind2[i] for i in idx2]
        # ind2 = ind2[:,idx2] ### reorder the flat cluster to match the order of the leaves the dendrogram
    if row_method != None:
        idx1 = Z1['leaves']  
        ### apply the clustering for the gene-dendrograms to the actual matrix data
        xt = xt[idx1, :]  # xt is transformed x
        # ind1 = ind1[idx1,:] ### reorder the flat cluster to match the order of the leaves the dendrogram
        ind1 = [ind1[i] for i in idx1]
    ### taken from http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python/3011894#3011894
    # print xt
    im = axm.matshow(xt, aspect='auto', origin='lower', cmap=cmap, norm=norm)  
    ### norm=norm added to scale coloring of expression with zero = white or black
    axm.set_xticks([])  ### Hides x-ticks
    axm.set_yticks([])

    # Add text
    new_row_header = []
    new_column_header = []
    for i in range(x.shape[0]):
        if row_method != None:
            if len(
                    row_header
            ) < 100:  ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1] - 0.5, i, '  ' + row_header[idx1[i]])
            new_row_header.append(row_header[idx1[i]])
        else:
            if len(
                    row_header
            ) < 100:  ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1] - 0.5, i,
                         '  ' + row_header[i])  ### When not clustering rows
            new_row_header.append(row_header[i])
    for i in range(x.shape[1]):
        if column_method != None:
            axm.text(
                i,
                -0.9,
                ' ' + column_header[idx2[i]],
                rotation=270,
                verticalalignment="top")  # rotation could also be degrees
            new_column_header.append(column_header[idx2[i]])
        else:  ### When not clustering columns
            axm.text(
                i,
                -0.9,
                ' ' + column_header[i],
                rotation=270,
                verticalalignment="top")
            new_column_header.append(column_header[i])

    for j in range(x.shape[0]):
        if row_method != None:
            axm.text(
                len(new_column_header) + 1,
                j,
                ' ' + row_header[idx1[j]],
                rotation=0,
                verticalalignment="top")  # rotation could also be degrees
            new_row_header.append(row_header[idx1[j]])
        else:  ### When not clustering columns
            axm.text(
                len(new_column_header) + 1,
                j,
                ' ' + row_header[j],
                rotation=0,
                verticalalignment="top")
            new_row_header.append(row_header[j])

    # Plot colside colors
    # axc --> axes for column side colorbar
    if column_method != None:
        axc = fig.add_axes([axc_x, axc_y, axc_w,
                            axc_h])  # axes for column side colorbar
        cmap_c = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        dc = np.array(ind2, dtype=int)
        dc.shape = (1, len(ind2))
        im_c = axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
        axc.set_xticks([])  ### Hides ticks
        axc.set_yticks([])

    # Plot rowside colors
    # axr --> axes for row side colorbar
    if row_method != None:
        axr = fig.add_axes([axr_x, axr_y, axr_w,
                            axr_h])  # axes for column side colorbar
        dr = np.array(ind1, dtype=int)
        dr.shape = (len(ind1), 1)
        #print ind1, len(ind1)
        cmap_r = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        im_r = axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
        axr.set_xticks([])  ### Hides ticks
        axr.set_yticks([])

    # Plot color legend
    axcb = fig.add_axes(
        [axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)  # axes for colorbar
    # print 'axcb', axcb
    cb = mpl.colorbar.ColorbarBase(
        axcb, cmap=cmap, norm=norm, orientation='horizontal')
    # print cb
    axcb.set_title("colorkey")

    cb.set_label("Differential Expression (log2 fold)")

    ### Render the graphic
    if len(row_header) > 50 or len(column_header) > 50:
        plt.rcParams['font.size'] = 14
    else:
        plt.rcParams['font.size'] = 18

    plt.show()


def compute_pca(X, verbose=False, **kwargs):
    """ 
        Compute PCA.
        :param X: Data 
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
        :return: Tuple (pca, X) containing a PCA object (see sklearn doc) and the transformed data
        :rtype: Tuple
    """
    pca = PCA(**kwargs)
    X = pca.fit_transform(X)

    print('Explained variance ratio of the {} components: \n {}'.format(pca.n_components_, 
                                                                        pca.explained_variance_ratio_))
    if verbose: 
        plt.bar(left=range(pca.n_components_), 
                height=pca.explained_variance_ratio_, 
                width=0.3, 
                tick_label=range(pca.n_components_))
        plt.title('Explained variance ratio by principal component')
        plt.show()
        
    return pca, X


def show_pca(X, y=None, i=1, j=2, verbose=False, **kwargs):
    """ 
        Plot PCA.
        :param X: Data
        :param y: Labels
        :param i: i_th component of the PCA
        :param j: j_th component of the PCA
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
    """
    pca, X = compute_pca(X, verbose, **kwargs)

    assert(i <= pca.n_components_ and j <= pca.n_components_ and i != j)

    if y is not None:
    
        if isinstance (y, pd.DataFrame):
            target_names = y.columns.values
            y = y.values
        elif isinstance(y, pd.Series):
            target_names = y.unique()
            y = y.values
        else:
            target_names = np.unique(y)

        if len(y.shape) > 1:
            if y.shape[1] > 1:
                y = np.where(y==1)[1]

        for label in range(len(target_names)):
            plt.scatter(X[y == label, i-1], X[y == label, j-1], alpha=.8, lw=2, label=target_names[label])
            
        plt.legend(loc='best', shadow=False, scatterpoints=1)
            
    else:
        plt.scatter(X.T[0], X.T[1], alpha=.8, lw=2)
            
    plt.xlabel('PC '+str(i))
    plt.ylabel('PC '+str(j))
    plt.title('Principal Component Analysis: PC{} and PC{}'.format(str(i), str(j)))
    plt.show()


def compute_lda(X, y, verbose=False, **kwargs):
    """ 
        Compute LDA.
        :param X: Data 
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for LDA (see sklearn doc)
        :return: Tuple (lda, X) containing a LDA object (see sklearn doc) and the transformed data
        :rtype: Tuple
    """
    lda = LinearDiscriminantAnalysis(**kwargs)
    X = lda.fit_transform(X, y)

    return lda, X


def show_lda(X, y, verbose=False, **kwargs):
    """ 
        Plot LDA.
        :param X: Data
        :param y: Labels
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
    """
    if isinstance (y, pd.DataFrame):
        target_names = y.columns.values
        y = y.values
    elif isinstance(y, pd.Series):
        target_names = y.unique()
        y = y.values
    else:
        target_names = np.unique(y)

    # Flatten one-hot
    if len(y.shape) > 1:
        if y.shape[1] > 1:
            y = np.where(y==1)[1]
    
    _, X = compute_lda(X, y, verbose=verbose, **kwargs)

    for label in range(len(target_names)):
        plt.scatter(X[y == label, 0], X[y == label, 1], alpha=.8, lw=2, label=target_names[label])

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset')
    plt.show()
    
def s2n(df):
    '''Perform signal to noise ration feature selection'''
    target = df.iloc[:,-1]
    columns = df.columns[:-1]
    s2n_coeff = pd.DataFrame(columns=['feat'], index=columns)
    for col in columns:
        mu1 = df[col][target==1].mean()
        mu2 = df[col][target==-1].mean()
        sigma1 = df[col][target==1].std()
        sigma2 = df[col][target==-1].std()
        coeff = (mu1-mu2)/(sigma1+sigma2)
        s2n_coeff['feat'][col] = coeff
    return s2n_coeff.astype('float64')

def check_na(df):
    '''Finds whether there are missing values.'''
    # I used this post https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
    na_columns=df.columns[df.isna().any()]
    found_na = df[df.isna().any(axis=1)][na_columns]
    print(found_na.head())
    return found_na

def shuffle(df, n=1, axis=0):  
    '''Shuffling after https://stackoverflow.com/questions/15772009/shuffling-permutating-a-dataframe-in-pandas'''
    df2 = df.copy()
    for _ in range(n):
        df2.apply(np.random.shuffle, axis=axis)
    return df2
     
def feature_learning_curve(data_df, sklearn_model, sklearn_metric):
    '''Run cross-validation on nested subsets of features generated by the Pearson correlated coefficient.'''
    print(sklearn_model.__class__.__name__.upper())
    corr = data_df.corr()
    sval = corr['fruit'][:-1].abs().sort_values(ascending=False)
    ranked_columns = sval.index.values
    print(ranked_columns) 
    result_df = pd.DataFrame(columns =["perf_tr", "std_tr", "perf_te", "std_te"])
    for k in range(len(ranked_columns)): 
        df = data_df[np.append(ranked_columns[0:k+1], 'fruit')]
        rdf =  pd.DataFrame(data=[df_cross_validate(df, sklearn_model, sklearn_metric)], columns =["perf_tr", "std_tr", "perf_te", "std_te"])
        result_df = result_df.append(rdf, ignore_index=True)
    return result_df
    
def svd_learning_curve(df, sklearn_model, sklearn_metric):
    '''Run cross-validation on features generated by SVD.'''
    print(sklearn_model.__class__.__name__.upper())
    df_scaled = (df-df.mean())/df.std()
    df_scaled = df_scaled.drop(columns=['fruit'])
    u, s, v = np.linalg.svd(df_scaled, full_matrices=True)
    labels= ['SV'+str(i) for i in range(1,df.shape[1])]
    fruit_name = ['Banana', 'Apple']
    fruit_list = [fruit_name[int((i+1)/2)] for i in df["fruit"].tolist()]
    #svd_df = pd.DataFrame(u[:,0:df_scaled.shape[1]], columns=labels)
    #svd_df = pd.DataFrame(u[:,0:3], columns=labels)
    #svd_df['fruit'] = fruit_list
    result_df = pd.DataFrame(columns =["perf_tr", "std_tr", "perf_te", "std_te"])
    for k in range(df_scaled.shape[1]): 
        svf_df = pd.DataFrame(u[:,0:k+1], columns=labels[0:k+1])
        svf_df['fruit'] = fruit_list
        rdf =  pd.DataFrame(data=[df_cross_validate(svf_df, sklearn_model, sklearn_metric)], columns =["perf_tr", "std_tr", "perf_te", "std_te"])
        result_df = result_df.append(rdf, ignore_index=True)
    return result_df
    
    
def save_images(all_files, filter = None, file_idx=None, file_type='z', dirname='./'):
    ''' Function that saves the images whose names are given in the list all_files. 
        Optionally provide a filter, which is a function to apply to the images.
        The filter takes a PIL image as input and returns either a PIL image or a numpy array.
        The files are re-ordered according to file_idx.'''
    if file_idx:
    	all_files = all_files[file_idx]
    k=1
    for filename in all_files:
    	s = "%s%02d.png" % (file_type, k)
    	# Filter the image
    	if filter:
        	img=get_image(filename)
        	img = filter(img)
        	img.save(os.path.join(dirname, s), 'PNG')
    	else:
        	cmd = "cp %s %s/%s" % (filename, dirname, s)
        	os.system(cmd)
    	k=k+1
    	