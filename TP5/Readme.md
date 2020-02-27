
## Deadline : Friday 13 March

# Table of content

* [Objective](#objective)
* [Preprocessing](#Preprocessing)
* [Model](#Model)
* [Visualization](#Visualization)
* 

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

1. include a preprocessor produced by the preprocessing team.
    - You can start using a sklearn preprocessor like KMeans or StandardScaler if the preprocessing team is not finished yet.
1. Run a cross-validated hyper parameter search on your model.
    - We expect you to save the measured performances of each cross validation iteration for each hyper-parameter you tried.
    - You may choose grid search or random search or any other seach.
    - Compare training and validation performances
    - work closely with the visualization team since they will help to visualize performances
1. Conclude on the tradeoff you may choose for your submission.

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

1. Look at the data
    - describe the distribution
1. Plot the performances of the cross-validated hyper-parameter search done by the Model team
    - You may use a fake one to start while the model team is working on it.
    - We expect plots with mean and standard variance of the performances (pyplot.errorbar) for each explored hyper parameter

