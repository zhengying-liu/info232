'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator

class model(BaseEstimator):
    ''' One Rule classifier '''
    def __init__(self):
        ''' The "constructor" initializes the parameters '''
        self.selected_feat = 0 	# The chosen variable/feature
        self.theta1 = 0 		# The first threshold
        self.theta2 = 0			# The second threshold
        self.is_trained=False

    def fit(self, X, Y, F=[]):
        ''' The method "fit" trains a super-simple classifier '''
        if not F: F=[str(item) for item in range(X.shape[1])]
        # First it selects the feature most correlated to the target
        correlations = np.corrcoef(X, Y, rowvar=0)
        self.selected_feat = np.argmax(correlations[0:-1, -1])
        best_feat = X[:, self.selected_feat]
        print('Feature selected = ' +  F[self.selected_feat])
        # Then it computes the average values of the 3 classes
        mu0 = np.median(best_feat[Y==0])
        mu1 = np.median(best_feat[Y==1])
        mu2 = np.median(best_feat[Y==2])
        # Finally is sets two decision thresholds
        self.theta1 = (mu0+mu1)/2.
        self.theta2 = (mu1+mu2)/2.
        self.is_trained=True

    def predict(self, X):
        ''' The method "predict" classifies new test examples '''
        # Select the values of the correct feature
        best_feat = X[:, self.selected_feat]
        # Initialize an array to hold the predicted values
        Yhat = np.copy(best_feat)				# By copying best_fit we get an array of same dim
        # then classify using the selected feature according to the cutoff thresholds
        Yhat[best_feat<self.theta1] = 0											# Class 0
        Yhat[np.all([self.theta1<=best_feat, best_feat<=self.theta2], 0)] = 1	# Class 1
        Yhat[best_feat>self.theta2] = 2 										# Class 2
        return Yhat

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
