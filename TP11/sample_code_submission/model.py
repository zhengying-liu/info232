"""
Created on Sat Mar 11 08:04:23 2017
Revised: Feb 2, 2019
Last revised: Apr 22, 2020

@author: isabelleguyon

This is an example of classifier program, we show how to combine
a classifier and a preprocessor with a pipeline. 

IMPORTANT: when you submit your solution to Codalab, the ingestion program 
should be able to find your model. 
It will load "model.py" from the sample_code/ directory. 
Your model class should be called "model".
"""

from sys import argv
from sys import path
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import numpy as np
    import pickle
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    from preprocessing import Preprocessor
    from os.path import isfile
    
class model (BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        #y = np.round(X[:,3])
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)

class RandomPredictor(BaseEstimator):
    ''' Make random predictions.'''	
    def __init__(self):
        self.target_num=0
        return
        
    def __repr__(self):
        return "RandomPredictor"

    def __str__(self):
        return "RandomPredictor"
	
    def fit(self, X, Y):
        if Y.ndim == 1:
            self.target_num=len(set(Y))
        else:
            self.target_num==Y.shape[1]
        return self
		
    def predict_proba(self, X):
        prob = np.random.rand(X.shape[0],self.target_num)
        return prob	
    
    def predict(self, X):
        prob = self.predict_proba(X)
        yhat = [np.argmax(prob[i,:]) for i in range(prob.shape[0])]
        return np.array(yhat)

class BasicClassifier(BaseEstimator):
    '''BasicClassifier: modify this class to create a simple classifier of
    your choice. This could be your own algorithm, of one for the scikit-learn
    classfiers, with a given choice of hyper-parameters.'''
    def __init__(self):
        '''This method initializes the parameters. This is where you could replace
        RandomForestClassifier by something else or provide arguments, e.g.
        RandomForestClassifier(n_estimators=100, max_depth=2)'''
        self.clf = RandomForestClassifier(random_state=1)

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def get_classes(self):
        return self.clf.classes_
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    
class MonsterClassifier(BaseEstimator):
    '''MonsterClassifier: This is a more complex example that shows how you can combine
    basic modules (you can create many), in parallel (by voting using ensemble methods)
    of in sequence, by using pipelines.'''
    def __init__(self):
        '''You may here define the structure of your model. You can create your own type
        of ensemble. You can make ensembles of pipelines or pipelines of ensembles.
        This example votes among two classifiers: BasicClassifier and a pipeline
        whose classifier is itself an ensemble of GaussianNB classifiers.'''
        fancy_classifier = Pipeline([
					('preprocessing', Preprocessor()),
					('classification', BaggingClassifier(base_estimator=GaussianNB(),random_state=1))
					])
        self.clf = VotingClassifier(estimators=[
					('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
					('Gaussian Classifier', GaussianNB()),
					('Support Vector Machine', SVC(probability=True)),
					('Fancy Classifier', fancy_classifier)],
					voting='soft')   
        
    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        return self.clf.fit(X, y)

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
        
def compute_accuracy(M, D, classifier_name):
	'''Evaluate the accuracy of M on D'''
	# Train
	Ytrue_tr = D.data['Y_train']
	M.fit(D.data['X_train'], Ytrue_tr)
    
	# Making classification predictions (the output is a vector of class IDs)
	Ypred_tr = M.predict(D.data['X_train'])
	Ypred_va = M.predict(D.data['X_valid'])
	Ypred_te = M.predict(D.data['X_test'])  
    
	# Training success rate and error bar:
	acc_tr = accuracy_score(Ytrue_tr, Ypred_tr)
        
	# Cross-validation performance:
	acc_cv = cross_val_score(M, D.data['X_train'], Ytrue_tr, cv=5, scoring='accuracy')
        
	print("%s\ttrain_acc=%5.2f(%5.2f)\tCV_acc=%5.2f(%5.2f)" % (classifier_name, acc_tr, ebar(acc_tr, Ytrue_tr.shape[0]), acc_cv.mean(), acc_cv.std()))
	# Note: we do not know Ytrue_va and Ytrue_te so we cannot compute validation and test accuracy
	return acc_tr


def test(D):  
    '''Function to try some examples classifiers'''    
    classifier_dict = {
            '1. MonsterClassifier': MonsterClassifier(),
            '2. SimplePipeline': Pipeline([('prepro', Preprocessor()), ('classif', BasicClassifier())]),
            '3. RandomPred': RandomPredictor(),
            '4. Linear Discriminant Analysis': LinearDiscriminantAnalysis()}
            
    for key in classifier_dict:
        myclassifier = classifier_dict[key]
        acc = compute_accuracy(myclassifier, D, key) # Replace by a call to ClfScatter
              
    return acc # Return the last accuracy (important to get the correct answer in the TP)
    
if __name__=="__main__":
    # We can use this function to test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../results"
        score_dir = "../scoring_program"
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        score_dir = argv[3]
                            
	# The M2 may have prepared challenges using sometimes AutoML challenge metrics
    path.append(score_dir)
    
    from zDataManager import DataManager # The class provided by binome 1
    
    basename = 'iris'
    D = DataManager(basename, input_dir) # Load data
    print(D)
    test(D)
 
