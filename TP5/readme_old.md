# /!\ Attention, la séance de la semaine prochaine consistera à écrire un rapport de proposition de projet /!\

# Ce TP n'aura donc pas lieu

## Deadline : Friday 13 March

# Table of content

* [Objective](#objective)
* [Preprocessing](#Preprocessing)
* [Model](#Model)
* [Visualization](#Visualization)

# Objective

- Free yourself from Notebooks !
- Write modules instead


# Preprocessing

Write some preprocessing class.

1. One to create new features
     - using clustering methods, hand written features, or anything else
1. One to reduce dimensions
    - PCA, SVD, feature selection or anything else
1. One to prepare data for the Classifier/Regressor
    - rescale or log scale if necessary
    - handle missing values
    - handle categorical data


If your preprocessing contains hyper-parameters or options they are expected to be set using key-word argument in the constructor. Like the `inplace` argument on the template bellow.

You do not need to copy all options from sklearn models. Only the ones relevant for your experiments.

## Template


Here is an atomic example of a Preprocessor class.
Change it to better correspond to your problem.


```python
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    FEATURE_LOG_IDX = (3, 7)
    def __init__(self, inplace=False):
        self.inplace = inplace
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        if not self.inplace:
            X = X.copy()
        X[FEATURE_LOG_IDX] = np.log(X[FEATURE_LOG_IDX])
        return X

```

# Model

Write 2 differents model class.

1. include one or 2 or all the preprocessors produced by the preprocessing team.
    - You can start using a sklearn preprocessor like KMeans or StandardScaler if the preprocessing team is not finished yet.
1. Run a cross-validated hyper parameter search on your model.
    - We expect you to save the measured performances of each cross validation iteration for each hyper-parameter you tried.
    - You may choose grid search or random search or any other seach.
    - Compare training and validation performances
    - work closely with the visualization team since they will help to visualize performances

If your model contains hyper-parameters or options they are expected to be set using key-word argument in the constructor. Like `n_estimators` argument on the template bellow.

You do not need to copy all options from sklearn models. Only the ones relevant for your experiments.

## Template

Here is an atomic example of a Model class.
Change it to better correspond to your problem.


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

class ModelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10):
        self.n_estimators
        # Preprocessing
        self.scaler = StandardScaler()
        # Classifier
        self.clf = GradientBoosting(n_estimators=n_estimators)


    def fit(self, X, y, sample_weights=None):
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        y_proba =  self.clf.predict_proba(X)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

```


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class ModelRegressor(BaseEstimator):
    def __init__(self, n_estimators=10):
        self.n_estimators
        # Preprocessing
        self.scaler = StandardScaler()
        # Regressor
        self.reg = GradientBoostingRegressor(n_estimators=n_estimators)

    def fit(self, X, y, sample_weights=None):
        X = self.scaler.fit_transform(X)
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.reg.predict(X)
        return y_pred

```

# Visualization

1. Help to describe the data after each preprocessing step with some visualization
    - distribution of features before/after log scale
    - t-SNE or other reduction to 2D to plot a subsample of the validation data. 
        - Are the target values (label or values) more separated in this 2D space ? You can use color for labels but also color shading or marksize for real values.
        - If you tried clustering before this preprocessing step do the same 2D plot with cluster label. Are the cluster still well separated ?
1. Plot the performances of the cross-validated hyper-parameter search done by the Model team
    - You may use a fake one to start while the model team is working on it.
    - We expect plots with mean and standard variance of the performances (pyplot.errorbar) for each explored hyper parameter
1. Try other relevant visualization to help your team read the results of experiments
    - confusion matrices
    - distribution plot of decision functions of classifier
    - distribution plot of regressor or classifier error
    - distribution of distances to cluster centers (help to find bondaries betweed clusters)
    - other things like colored table for hyper parameters performances
    
    


# Submit

Submit your work as a notebook containing main sections with :

1. Model description
    - Quickly describe the 3 preprocessors you built and explain why do you think they can help your model.
        - We do not need the preprocessing code here
        - we explect the descriptions to be smaller than the code itself
    - Describe your full model
        - in which order the preprocessing are made and why this order was chosen
        - why did you choose this classifier/regressor
        - how many hyper-parameter does your full model require (including preprocessor hyper parameters) ?
1. Data exploration
    - For each preprocessing show the plots you have made to help describing the data
        - we do not need your code. 
        You can import your plot functions from the python modules you have writen !
        - Give some insight or some remarks on those plots. What do you see ? What can you conclude ? Are you disapointed or surpised by the results ?
        
1. Hyper parameter search
    - Quickly explain your choice of search algorithm
    - Quickly explain your choice of hyper-parameter to tune
    - For all hyper paramter
        - Describe performances
        - then give a hypothesis to explain the observed performances
        - then challenge this hypothesis and test it more if possible
        - Are you disapointed or surpised by the results ? Why ?
    - Conclude on the tradeoff you may choose for your submission.

We encourage you to explore other ideas and give more insights if you can !

## Grading

We use a very complex algorithm but open source algorithm :



```python
import numpy as np
grade = np.random.randint(0, 21)
```

But also use :

- 1 point per wanted classes (total = 5 points) 
    - 3 Preprocessing classes
    - 2 Models classes
- 5 points for visualization to help
    - 1 for hyper-parameter
    - 1 for 2D plots
    - 1 for each other relevant visualizations
        - confusion matrices
        - etc
    
- 10 points for the quality of the report and its contained analysis. Still uncertain on how those points will be scatered between :
    - Preprocessing summary
    - Full Model description
    - Data exploration
    - Hyper parameter search
