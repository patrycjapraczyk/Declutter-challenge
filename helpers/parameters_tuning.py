import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


from models.model_exec import ModelExec

model_exec = ModelExec(include_comments=False, include_long_code=True)
features = model_exec.extract_features(model_exec.data)
features = model_exec.combine_features(features, include_comments=False)
x = features
y = model_exec.data['non-information']


# import warnings
# warnings.filterwarnings('ignore')


def tune_logistic_regression(x, y):
    grid = {
        'C': np.logspace(-4, 4, 50),
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    }  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("accuracy :", logreg_cv.best_score_)


def tune_ada_boost(x, y):
    grid = {
        'n_estimators': [165, 169, 170, 171, 172, 175, 180],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        # 'algorithm': ['SAMME', 'SAMME.R']
        'random_state': range(0, 42)
    }
    model = AdaBoostClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_SVM(x, y):
    grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.001, 0.0001],
        'kernel': ['linear', 'rbf'],
        # 'random_state': 42,
    }
    model = SVC()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_ada_boost(x, y):
    grid = {
        'n_estimators': [165, 169, 170, 171, 172, 175, 180],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        # 'algorithm': ['SAMME', 'SAMME.R']
        'random_state': range(0, 42)
    }
    model = AdaBoostClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_random_forest(x, y):
    grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    model = RandomForestClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_decision_trees(x, y):
    grid = {
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
        'min_samples_split': [0.1, 1, 2, 5, 8, 10],
        'max_depth': [None, 2, 3, 4, 5, 6, 80, 90, 100, 110],
        'criterion': ["gini", "entropy"],
        'min_weight_fraction_leaf': [0, 0.01, 0.1]
    }
    model = DecisionTreeClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_MLP(x, y):
    grid = {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    model = MLPClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_gradient_boosting(x, y):
    grid = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [None, 2, 3, 4, 5, 6, 80, 90, 100, 110],
        'min_samples_split': [100, 200, 400, 600],
        'subsample': [1, 0.8, 0.9, 0.7, 0.6],
        'warm_start':[True, False],
        'min_samples_leaf': [20, 50, 80],

    }
    model = GradientBoostingClassifier(n_estimators=660)
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_kneighbours(x, y):
    grid = {
        'n_neighbors': np.arange(1, 16),
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'weights': ['uniform', 'distance'],
        'leaf_size': [10, 20, 30, 40, 50, 60]
    }
    model = KNeighborsClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


def tune_naive_bayes(x, y):
    grid = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    model = KNeighborsClassifier()
    model_cv = GridSearchCV(model, grid, n_jobs=-1, cv=10, verbose=10)
    model_cv.fit(x, y)

    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    print("accuracy :", model_cv.best_score_)


########################
## COMMENT DATA

model_exec = ModelExec(include_comments=False, include_long_code=True)
comments = model_exec.data['comment']
comments = model_exec.vectorise_comment_data(comments, comments)
x = comments[0]
y = model_exec.data['non-information']


tune_ada_boost(x, y)
