# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


def answer_01(data):
    """
    Write a code which standardizes the column of given data EXCEPT THE LAST ONE !

    Args
    ----
        data [pandas.DataFrame]: the data which should be standardized

    Returns
    -------
        scaled_data [pandas.DataFrame]: Rescaled data except last column

    """
    # TODO : standardize data
    scaled_data = data
    return scaled_data


def answer_02():
    """
    Import KNeighborsClassifier from the scikit-learn library 

    Returns
    -------
        nearest_neighbors class
    """
    # Wrong classifier
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB


def answer_03():
    """
    Import balanced_accuracy_score from scikit-learn

    Args
    ----

    Returns
    -------
    """
    from sklearn.metrics import accuracy_score as sklearn_metric
    return sklearn_metric


def answer_04():
    """
    In this context :
    Does re-scaling variables always help ?

    Returns
    -------
        YES or NO

    """
    YES = 'YES'
    NO = 'NO'
    # Return YES or NO
    return None


def answer_05():
    """
    In which case does it help most?

    Returns
    -------
     CASE1, CASE2, CASE3 or CASE4
    """
    CASE1 = "RAW"
    CASE2 = "RE"
    CASE3 = "CS"
    CASE4 = "CROP"
    # Return CASE1, CASE2, CASE3 or CASE4
    return None



def answer_06():
    """
    If the test performance is bad but the training performance is good,
     is the model under-fitting or over-fitting? 
    """
    under_fitting = "under-fitting"
    over_fitting  = "over-fitting"
    # Return under_fitting or over_fitting
    return None


def answer_07():
    """
    If both are bad, is the model is under-fitting or over-fitting ? 
    """
    under_fitting = "under-fitting"
    over_fitting  = "over-fitting"
    # Return under_fitting or over_fitting
    return None


def answer_08():
    """
    Which models are over-fitting ?
    """
    model_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    overfitting_models = ["Replace", "with", "model", "names", "from", "the",  "list", "above"]
    return overfitting_models


def answer_09():
    """
    Which models are are under-fitting ?
    """
    model_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    underfitting_models = ["Replace", "with", "model", "names", "from", "the", "list", "above"]
    return underfitting_models





