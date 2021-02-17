from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

def logistic_grid():

    return LogisticRegression(), {'C':[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                  'penalty':['l1'],
                                  'solver':['liblinear'],
                                  'max_iter': [100],
                                  }

def svc_grid():
    return LinearSVC(), { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'penalty': ['l2'],
                    #'penalty': ['l1'], 'dual': [False],
                    'max_iter': [1500],
                    }


def randomforest_grid():
    return RandomForestClassifier(), {'n_estimators': [20, 50],
                                      'max_features': [50, 0.5, None],
                                      'max_depth': [3, None],
                                      'criterion': ['gini'],
                                    }

def svm_grid():
    from sklearn.svm import SVC
    return SVC(), { 'C': [0.01, 0.1, 1, 10, 100],
                    'gamma': ['auto', 0.01, 0.1, 1, 10, 100]}

def build_classifier(c):
    '''
    Factory method building scikit-learn classifiers.
    :param c: classifier label
    '''
    if (c.endswith('-grid')): # create grid for cross valid grid search
        classif = c[:-5]
        if classif == 'logreg': return logistic_grid()
        elif classif == 'svc': return svc_grid()
        elif classif == 'rf': return  randomforest_grid()
        elif classif == 'svm': return svm_grid()
    else:
        if c == 'logreg':
            return LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1', C=1.0)
        elif c == 'svc':
            return SVC()