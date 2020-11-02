from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def random_search():
    # RANDOM SEARCH
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    if False:
        # RANDOM SEARCH
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)

    base_model = RandomForestRegressor(n_estimators=100, random_state = 42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


def evaluate(model, test_features, test_labels):
    r2_score = model.score(test_features, test_labels)
    print('Model Performance')
    print('R2 Score: {:0.4f}'.format(r2_score))

    return r2_score



def gridsearch_cv():
    param_grid = {
        'max_depth': [50, 75, 100],
        'max_features': [20, 50, 70, 'auto'],
        'n_estimators': [200, 400, 600]
    }
    # Create a based model
    rf = RandomForestRegressor(oob_score=True)

    if False:
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                                  cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)
