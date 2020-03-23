import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from configparser import ConfigParser
import json, os, ast, logging, joblib

def extract_params(file_name):
    """
    Function to extract parameters from .py file
    :param file_name: file name without .py
    :return: dictionary of parameters
    """
    # read params
    parser = ConfigParser()
    parser.read('./mlp/params/'+file_name+'.py')
    params = {}
    for section in parser.sections():
        indv_params = {}
        for option in parser.options(section):
            indv_params[option] = [i for i in ast.literal_eval(parser.get(section, option))]
        params[section] = indv_params
    return params

# randomized gridsearch CV
# store validation results and best estimators
def grid_search(models, params, X_train, y_train, n_splits=3, n_iter=5):
    """
    Function for randomsearch CV to get best estimators
    :param models: dictionary of models to be used
    :param params: dictionary of models' parameters to be searched
    :param X_train: training explanatory features
    :param y_train: training target feature
    :param n_splits: number of cross validation to be used
    :param n_iter: number of parameter combinations to iterate
    :return: best estimators from the grid search
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # no. of crossvalidation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_estimators = {}
    for key, model in models.items():
        print(key, ' search')
        try:
            # gridsearch to get best parameters
            model_param_search = RandomizedSearchCV(estimator=model, param_distributions=params[key],
                                                    scoring='neg_mean_squared_error',
                                                    n_jobs=4, verbose=0, random_state=42, n_iter=n_iter,
                                                    cv=[(train_split_index, val_index) for train_split_index, val_index in
                                                        tscv.split(X_train)])

            model_param_search.fit(X_train, y_train.reshape(len(y_train), ))
            best_estimators[key] = model_param_search.best_estimator_

        except Exception as e:
            logger.error(f'Training for {key} failed: ', str(e))

    return best_estimators

#import datasets
X_train = np.load('./mlp/process_data/X_train.npy')
y_train = np.load('./mlp/process_data/y_train.npy')
X_test = np.load('./mlp/process_data/X_test.npy')
y_test = np.load('./mlp/process_data/y_test.npy')

#read params
register_params = extract_params('register_params_search')
guest_params = extract_params('guest_params_search')

#instantiate models
xgbr = XGBRegressor()
lassor = Lasso()
rfr = RandomForestRegressor()
models = {'xgbr':xgbr,'lassor':lassor,'rfr':rfr}

#gridsearch CV
register_optimised_estimators = grid_search(models, register_params, X_train[24:,:], y_train[24:,5], n_splits=2, n_iter=2) #for former iterations n_splits=3, n_iter=5
guest_optimised_estimators = grid_search(models, guest_params, X_train[24:,:], y_train[24:,6], n_splits=2, n_iter=2) #for former iterations n_splits=3, n_iter=5

for key, model in models.items():
    # print('register:', register_optimised_estimators[key])
    # print('guest:', guest_optimised_estimators[key])
    joblib.dump(register_optimised_estimators[key], './mlp/optimised_models/register_opt_'+str(key)+'.pkl')
    joblib.dump(guest_optimised_estimators[key], './mlp/optimised_models/guest_opt_' + str(key) + '.pkl')
print('Gridsearch completed, models stored in optimised_models file.')