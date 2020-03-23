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


def convert_y(y_df, y_pred, user='reg_user'):
    """
    Function to convert predicted log user difference into number of users
    :param y_df: dataframe consisting of guest and registered user numbers, log values, difference in log values
    :param y_pred: predicted y array
    :param user: a string - 'reg_user' for registered user or 'gs_user' for guest user
    :return: y_df2 consist of y_df and the concatenated converted number of guest and registered users
    """
    y_df2 = pd.concat([y_df, pd.DataFrame(y_pred)], axis=1)
    y_df2.columns = y_df.columns.tolist() + ['pred_diff_lg_' + user]
    y_df2['pred_lg_' + user] = y_df2['pred_diff_lg_' + user] + y_df2['lg_' + user]
    y_df2['pred_' + user] = y_df2['pred_lg_' + user].apply(lambda x: np.exp(x) - 0.1)
    y_df2['pred_' + user] = y_df2['pred_' + user].apply(lambda x: 0 if x < 0 else np.round(x)).shift(24)

    return y_df2


def metrics_scores(y_df_pred, actual='registered_users', pred='pred_reg_user'):
    """
    Function to collate metrics of prediction score for user type
    :param y_df_pred: dataframe consisting of actual and predicted guest / registered user numbers
    :param actual: string - 'registered_users' to retrieve actual registered user or 'guest_user' to retrieve actual guest user from dataframe
    :param pred: string - 'pred_reg_user' to retrieve predicted registered user or 'pred_gs_user' to retrieve predicted guest user from dataframe
    :return: metrics consist of r-square, rmse and rmse to target mean ratio
    """
    metrics = {}
    metrics['r2'] = np.round(r2_score(y_df_pred[actual][24:], y_df_pred[pred][24:]), 3)
    metrics['rmse'] = np.round(np.sqrt(mean_squared_error(y_df_pred[actual][24:], y_df_pred[pred][24:])), 2)
    metrics['y_mean'] = np.round(np.mean(y_df_pred[actual]), 2)
    metrics['rmse_to_y_mean'] = np.round(
        np.sqrt(mean_squared_error(y_df_pred[actual][24:], y_df_pred[pred][24:])) / np.mean(y_df3[actual]), 2)

    return metrics


## Get processed dataset
X_train = np.load('./mlp/process_data/X_train.npy')
y_train = np.load('./mlp/process_data/y_train.npy')
X_test = np.load('./mlp/process_data/X_test.npy')
y_test = np.load('./mlp/process_data/y_test.npy')
X_col_names = np.load('./mlp/process_data/X_col_names.npy', allow_pickle=True)
y_col_names = np.load('./mlp/process_data/y_col_names.npy', allow_pickle=True)

## Get optimised models
register_opt_estimators = {}
guest_opt_estimators = {}

# Get selected optimised models to be evaluated and tested
# get selected model names
parser = ConfigParser()
parser.read('./mlp/optimised_models/opt_models_select.py')
for section in parser.sections():
    model_nm_dict = {}
    for option in parser.options(section):
        model_nm_dict[option] = [i for i in ast.literal_eval(parser.get(section, option))]

# retrieve the selected models
for key, name in model_nm_dict.items():
    register_opt_estimators[key] = joblib.load('./mlp/optimised_models/register_opt_' + str(key) + '.pkl')
    guest_opt_estimators[key] = joblib.load('./mlp/optimised_models/guest_opt_' + str(key) + '.pkl')

## Cross validation metrics
# get number of cv splits
tscv = TimeSeriesSplit(n_splits=3)
counter = 0
# dictionary to store cv metrics in
cv_metrics_dict = {}
user_pred_dict = {'registered_users': 'pred_reg_user',
                  'guest_users': 'pred_gs_user',
                  # 'target': 'pred_target'
                  }
## Run cross_validation
for train_split_index, val_index in tscv.split(X_train):
    X_train_cv = X_train[train_split_index].copy()
    y_train_cv = y_train[train_split_index].copy()
    X_val_cv = X_train[val_index].copy()
    y_val_cv = y_train[val_index].copy()
    counter += 1
    cv_metrics_dict[counter] = {}
    for key, name in model_nm_dict.items():
        # cv for registered users
        # fit on training dataset
        register_opt_estimators[key].fit(X_train_cv[24:, :].copy(),
                                         y_train_cv[24:, 5].copy().reshape(len(y_train_cv[24:, 5]), ))
        # evaluate model on validation dataset
        reg_user_pred = register_opt_estimators[key].predict(X_val_cv[24:, :].copy())

        # convert prediction back to user number
        y_df = pd.DataFrame(y_val_cv)
        y_df.columns = y_col_names
        y_df2 = convert_y(y_df, reg_user_pred, user='reg_user')

        # cv for guest users
        # fit on training dataset
        guest_opt_estimators[key].fit(X_train_cv[24:, :].copy(),
                                      y_train_cv[24:, 6].copy().reshape(len(y_train_cv[24:, 6]), ))
        # evaluate model on validation dataset
        guest_user_pred = guest_opt_estimators[key].predict(X_val_cv[24:, :].copy())

        # convert prediction back to user number
        y_df3 = convert_y(y_df2, guest_user_pred, user='gs_user')

        # store evaluation metrics
        cv_metrics_dict[counter][key] = {}
        for actual, pred in user_pred_dict.items():
            cv_metrics_dict[counter][key][actual] = metrics_scores(y_df3, actual=actual, pred=pred)

## Convert metrics into dataframe to sort for best registered/guest user model
cv_metrics_df = pd.DataFrame.from_dict({(i, j, k): cv_metrics_dict[i][j][k]
                                        for i in cv_metrics_dict.keys()
                                        for j in cv_metrics_dict[i].keys()
                                        for k in cv_metrics_dict[i][j].keys()},
                                       orient='index').reset_index()
cv_metrics_df.columns = ['cv', 'model', 'user', 'r2', 'rmse', 'y_mean', 'rmse_to_y_mean']
cv_agg_metrics = cv_metrics_df.groupby(['user', 'model']).mean().sort_values(['user', 'r2'], ascending=False).drop('cv', axis=1).reset_index()
cv_agg_metrics.to_csv('./mlp/output/cv_agg_metrics.csv')
print('Models have been evaluated, please check metrics in output file "cv_agg_metrics.csv".')
# print(cv_agg_metrics)

## Select best models for test set.
best_reg_user_model = cv_agg_metrics[cv_agg_metrics['user'] == 'registered_users']['model'].head(1).values[0]
best_gs_user_model = cv_agg_metrics[cv_agg_metrics['user'] == 'guest_users']['model'].head(1).values[0]
print('Select registered user best model:', model_nm_dict[best_reg_user_model], ', guest user best model:', model_nm_dict[best_gs_user_model], 'for test.')

# fit registered user model and predict
register_opt_estimators[best_reg_user_model].fit(X_train[24:, :].copy(),
                                                 y_train[24:, 5].copy().reshape(len(y_train[24:, 5]), ))
reg_user_pred = register_opt_estimators[best_reg_user_model].predict(X_test[24:, :].copy())

# convert registered user log diff prediction back to user number
y_df = pd.DataFrame(y_test)
y_df.columns = y_col_names
y_df2 = convert_y(y_df, reg_user_pred, user='reg_user')

# fit guest user model and predict
guest_opt_estimators[best_gs_user_model].fit(X_train[24:, :].copy(),
                                             y_train[24:, 6].copy().reshape(len(y_train[24:, 6]), ))
guest_user_pred = guest_opt_estimators[best_gs_user_model].predict(X_test[24:, :].copy())

# convert guest user log diff prediction back to user number
y_df3 = convert_y(y_df2, guest_user_pred, user='gs_user')

# get test metrics output
y_df3['pred_target'] = y_df3['pred_gs_user'] + y_df3['pred_reg_user'] #aggregate predicted users into predicted target
test_metrics = metrics_scores(y_df3, actual='target', pred='pred_target')
test_metrics_df = pd.DataFrame(test_metrics, index=['test_scores'])
test_metrics_df.to_csv('./mlp/output/test_metrics.csv')

# get model output
joblib.dump(register_opt_estimators[best_reg_user_model],
            './mlp/output/register_best_model_' + str(best_reg_user_model) + '.pkl')
joblib.dump(guest_opt_estimators[best_gs_user_model],
            './mlp/output/guest_best_model_' + str(best_gs_user_model) + '.pkl')
print('Best models are saved and results are stored in output file "test_metrics.csv".')