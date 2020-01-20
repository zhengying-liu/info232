# Load general libraries
import os, re
from glob import glob as ls
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_files(datadir, type = 'a'):
    ''' Function that gets all the files as a list. '''
    file_list = list()
    for root, dirs, files in os.walk(datadir):
        for name in files:
            if not re.search('.ipynb|.DS_Store', name) and re.search(type, name):
                file_list.append(os.path.join(root, name))
    return sorted(file_list, key = str)

def get_image(path):
    ''' Function that gets an image and set its background to transparent. '''
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

def show_images(all_files, filter = None, columns = 5, file_idx=None):
    ''' Function that shows the images whose names are given in the list all_files. 
        Optionally provide a filter, which is a function to apply to the images.
        The filter takes a PIL image as input and returns either a PIL image or a numpy array.
        The files are re-ordered according to file_idx.'''
    if file_idx:
    	all_files = all_files[file_idx]
    rows = len(all_files)/columns
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
        k=k+1
 
def preprocess_data(data_dir, extract_features, standardize=False, verbose = False):      
    ''' Function that preprocesses all files and returns an array X with examples in lines
    	and features in column, and a column array Y with truth values.'''
    # Read and convert a_files
    a_files = get_files(data_dir, 'a')
    n = len(a_files)
    _X = np.zeros([2*n, 2])
    Y = np.zeros([2*n, 1])
    for i in range(n):
    	if verbose: print(a_files[i])
    	img=get_image(a_files[i])
    	_X[i, :] = extract_features(img, verbose)
    	Y[i] = 1 # Apples are labeled 1
    	
    # Read and convert b_files
    b_files = get_files(data_dir, 'b')
    n = len(b_files)
    for i in range(n):
    	if verbose: print(b_files[i])
    	img=get_image(b_files[i])
    	_X[n+i, :] = extract_features(img, verbose)
    	Y[n+i] = -1 # Bananas are labeled -1
    	
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
	fig=plt.figure(figsize=(5,10))
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
			if filter:
				img = filter(img)
			ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
			ax.add_artist(ab)
	else: 	# Plot UNLABELED test examples
		f = F[test_index]
		ax.scatter(x, y, s=250, marker='s', c='k') 
		ax.scatter(x, y, s=100, marker='$?$', c='w') 
   	
	if axis == 'square':			
		ax.set_aspect('equal', adjustable='box')
	plt.xlim(-3, 4)
	plt.ylim(-2, 3)
	plt.xlabel('$x_1$ = Redness')
	plt.ylabel('$x_2$ = Elongation')
	
	# Add line on the diagonal
	if show_diag:
		plt.plot([-3, 3], [-3, 3], 'k--')
	return
	
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    ul = unique_labels(y_true, y_pred)
    if np.sum(ul)==0:
    	ul = [0, 1]
    classes = classes[ul]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
