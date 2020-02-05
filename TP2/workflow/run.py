
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

def main():
    N_CV = 3
    RESULT_PATH = 'results.csv'
    clf = KNeighborsClassifier()

    cross_val = ShuffleSplit(n_splits=N_CV, test_size=0.4)

    all_monitors = []

    for i_cv, (train_idx, test_idx) in enumerate(cross_val.split(y)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test  = X[test_idx]
        y_test  = y[test_idx]

        monitor = {}
        monitor['i_cv'] = i_cv

        clf.fit(X_train, y_train)

        accuracy = clf.score(X_train, y_train)
        monitor['train_accuracy'] =  accuracy

        accuracy = clf.score(X_test, y_test)
        monitor['test_accuracy'] =  accuracy

        all_monitors.append(monitor.copy())
    resultats = pd.DataFrame(all_monitors)
    resultats.to_csv(RESULT_PATH)
