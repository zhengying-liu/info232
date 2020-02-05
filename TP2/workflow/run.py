
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from datasets import load_data

def parse_args():
    pass

def choose_clf(name):
    theclass = ALL_MODEL_CLASSES[name]
    clf = theclass()
    return clf

def main():
    args = parse_args()

    print('loading data ...')
    X, y = load_data()

    N_CV = 3
    RESULT_PATH = 'results.csv'

    print('set up model and cross validation ...')
    clf = choose_classifier(args)

    cross_val = ShuffleSplit(n_splits=N_CV, test_size=0.4)

    all_monitors = []

    for i_cv, (train_idx, test_idx) in enumerate(cross_val.split(y)):
        print('starting iter', i_cv, flush=True)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test  = X[test_idx]
        y_test  = y[test_idx]

        monitor = {}
        monitor['i_cv'] = i_cv

        clf.fit(X_train, y_train)

        monitor = evaluate_on_train(clf, X_test, y_test, monitor)
        monitor = evaluate_on_test(clf, X_test, y_test, monitor)

        all_monitors.append(monitor.copy())
    print('saving results to', RESULT_PATH)
    resultats = pd.DataFrame(all_monitors)
    resultats.to_csv(RESULT_PATH)
    print('DONE !')


def evaluate_on_test(clf, X_test, y_test, monitor):
    _evaluate_on_data(clf, X_test, y_test, monitor, prefix)
    score = balanced_score(clf, X_test, y_test)
    monitor["train balanced_score"
    return monitor

def evaluate_on_train(clf, X_train, y_train, monitor):
    _evaluate_on_data(clf, X_train, y_train, monitor, prefix='train')
    return monitor

def _evaluate_on_data(clf, X, y, monitor):
    accuracy = clf.score(X, y)
    monitor[prefix+'_accuracy'] =  accuracy
    
    confusion_matrix = ...
    monitor['True_positif'] = confusion_matrix[0,0]
    monitor['True_negativ'] = confusion_matrix[0,1]
    monitor['false_positif'] = confusion_matrix[1,0]
    monitor['false_positif'] = confusion_matrix[1,1]
    return monitor


if __name__ == "__main__":
    main()




