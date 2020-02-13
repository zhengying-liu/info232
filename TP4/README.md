Instructions for TP4
========

This week you will work on improving the starting kit in order to have better performance. Each pair of student will be assigned to specific task (pre-processing, visualizaton, model) !

The [teams](http://saclay.chalearn.org/home/teams_l2_2019_2020) and the [projects](http://saclay.chalearn.org/) can be found on the course's website.

_Take a moment to navigate through your challenge website to get familiar with the interface and read a bit about the problem..._



Table of Contents
=================
* [Step 1: Subgroup and environment setup](#step-1-Subgroup-and-environment-setup)
* [Step 2: Tasks](#step-2-tasks)
* [Step 3: Push your changes](#step-3-Push-your-changes)

## Step 1: Subgroup and environment setup

### 1.1 Define subgroup

For each team, create subgroups of student for the three tasks: **pre-processing**, **model** and **visualization**.

**GROUP LEADER:** By replying your welcome e-mail, send us the name of each student pairs with their corresponding task and the link of your team Github repository.

### 1.2 Environment setup

**EVERYONE IN THE GROUP, INCLUDING THE GROUP LEADER:**

In order to prevent conflict changes on git repository, each pairs of student should copy the original `README.ipynb` and work on their own copy.


```bash
cd ~/projects
cd groupname # /!\ REPLACE groupname by your groupname /!\
cd starting_kit
cp README.ipynb README_{YOUR-SUBGROUP}.ipynb # YOUR-SUBGROUP: preprocessing, visualizaton, model
```


**GROUP LEADER:**
Add each member as a collaborator into the github repository (github repository -> Settings -> Manage access).


## Step 2: Tasks

You are supposed to work in pair. Open the notebook (copy version, `README_{YOUR-SUBGROUP}.ipynb`) and follow instruction depending on your task. Your grade will depend on the readability of the notebook, quality of your code and analysis of the result.


### Pre-processing subgroup

For a general introduction of frequently used method, you can refer to [sklearn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).

Mandatory questions are:

* Detect if there are [outliers](https://scikit-learn.org/stable/modules/outlier_detection.html) in the data. Propose method to filter / fix outliers if necessary.

* Apply dimension reduction ([PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) or [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)) method and observe final performance when varying the number of dimension.

* Try [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) (remove unnecessary feature in the data) and observe performance. Then highlight most important features.

**Bonus**: Generate new features and see if performance improves



### Model subgroup

Your goal is the find the most performing machine learning model (and its corresponding hyperparameters) for your problem.

A (non exhaustive) list of machine learning model can be found [here](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning).

Mandatory questions are:

* Choose and test 5 models using cross-validation and report their performances. You can refer to TP2.

* Read hyper-parameter documentation of the best performing model and write a small report (on the notebook) describing hyper-parameter importance, its influence on the training time and the risk of over-fitting.

* Find best hyper-parameters (of the best model) with [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) or [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV).

**Bonus**: Try ensemble learning by combining prediction from different model ([stacking](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization), [voting](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier), ...).


### Visualizaton subgroup

Your goal is to create useful visualization. We encourage you to look at these [Seaborn examples](https://seaborn.pydata.org/examples/index.html).


Mandatory questions are:

* Plot confusion matrix: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

* Create interesting visualization to investigate error in prediction: classifier ([decision boundary](https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html)), regressor ([residual error](https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html))

* Plot performance of machine learning model with error (see TP2).

**BONUS**: Visualize cluster in your data.



## Step 3: Push your changes

**Everyone:** You should update your team repository.

```bash
git add README_{YOUR-SUBGROUP}.ipynb # YOUR-SUBGROUP: preprocessing, visualizaton, model
git commit -m "improve starting kit: {YOUR-SUBGROUP}"
git pull
git push
```
