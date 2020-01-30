# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from teacher import Score
from teacher import SubQuestion

import os
import pandas as pd
import numpy.testing as npt



def _load_data():
    data_dir = './TP2/mini-dataset/'
    data = pd.read_csv(os.path.join(data_dir, 'RE_data.csv'))
    return data


@Score(1)
@SubQuestion('question_01')
def question_01_a(answer):
    """
    Write a code which standardizes the column of given data EXCEPT THE LAST ONE !

    Args
    ----
        data [pandas.DataFrame]: the data which should be standardized

    Returns
    -------
        scaled_data [pandas.DataFrame]: Rescaled data except last column

    """
    data = _load_data()
    scaled_data = answer(data)
    col_0_mean, col_1_mean, col_2_mean = scaled_data.mean()
    correct_col_0_mean = -1.0853504110388495e-16
    correct_col_1_mean = -8.266752297412571e-16
    correct_col_2_mean = 0.3564154786150713
    npt.assert_almost_equal(col_0_mean, correct_col_0_mean, 
                            err_msg='hint : problem in column 0', verbose=False)
    npt.assert_almost_equal(col_1_mean, correct_col_1_mean,
                            err_msg='hint : problem in column 1', verbose=False)
    npt.assert_almost_equal(col_2_mean, correct_col_2_mean,
                            err_msg='hint : problem in last column', verbose=False)

@Score(1)
@SubQuestion('question_01')
def question_01_b(answer):
    """
    Write a code which standardizes the column of given data EXCEPT THE LAST ONE !

    Args
    ----
        data [pandas.DataFrame]: the data which should be standardized

    Returns
    -------
        scaled_data [pandas.DataFrame]: Rescaled data except last column

    """
    data = _load_data()
    scaled_data = answer(data)
    col_0_std, col_1_std, col_2_std = scaled_data.std()
    correct_col_0_std = 1.
    correct_col_1_std = 1.
    correct_col_2_std = 0.9352804787677323
    npt.assert_almost_equal(col_0_std, correct_col_0_std, 
                            err_msg='hint : problem in column 0', verbose=False)
    npt.assert_almost_equal(col_1_std, correct_col_1_std,
                            err_msg='hint : problem in column 1', verbose=False)
    npt.assert_almost_equal(col_2_std, correct_col_2_std,
                            err_msg='hint : problem in last column', verbose=False)


@Score(1/4)
def question_02(answer):
    """
    Import KNeighborsClassifier from the scikit-learn library 
    """
    from sklearn.neighbors import KNeighborsClassifier
    some_class = answer()
    assert some_class is KNeighborsClassifier



@Score(1/4)
def question_03(answer):
    """
    import the balanced_accuracy_score
    """
    from sklearn.metrics import balanced_accuracy_score as sklearn_metric
    some_class = answer()
    assert some_class is sklearn_metric



@Score(1/4)
def question_04(answer):
    """
    In this context :
    Does re-scaling variables always help ?

    """
    a = answer()
    assert a == 'YES'


@Score(1/4)
def question_05(answer):
    """
    In this context :
    In which case does it help most?


    """
    a = answer()
    assert a == 'CS'




@Score(1/2)
def question_06(answer):
    """
    If the test performance is bad but the training performance is good,
     is the model under-fitting or over-fitting? 

    """
    a = answer()
    assert a == "over-fitting"


@Score(1/2)
def question_07(answer):
    """
    If both are bad, is the model is under-fitting or over-fitting ? 

    """
    a = answer()
    assert a == "under-fitting"


@Score(1/2)
def question_08(answer):
    """
    Which models are over-fitting ?

    """
    a = answer()
    assert a == ['RBF SVM', 'Decision Tree', 'Random Forest']


@Score(1/2)
def question_09(answer):
    """
    Which models are under-fitting ?

    """
    a = answer()
    assert a == ['Linear SVM', 'Naive Bayes']



