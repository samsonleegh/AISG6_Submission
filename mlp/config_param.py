"""
Change hyperparameters of models for gridsearch
"""
from configparser import ConfigParser

##################################################
##get configurations for models on registered users
register_config = ConfigParser()

# 1st iteration wide search
# register_config['xgbr'] = {'min_child_weight' : [5,10,15],
#                     'objective' : ['reg:squarederror'],
#                     'gamma' : [0.1, 0.5, 0.8],
#                     'learning_rate' : [0.3, 0.1, 0.01, 0.001],
#                     'max_depth' : [5,10,20],
#                     'subsample' : [0.1,0.5,0.8],
#                     'n_estimators' : [100,500,1000]}

# register_config['lassor'] = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                     'selection' : ['cyclic', 'random']}

# register_config['rfr'] = {'bootstrap' : [True, False],
#                 'n_estimators' : [100,500,1000],
#                 'max_features' : ['sqrt', 'log2'],
#                 'min_samples_split' : [5, 10, 20],
#                 'min_impurity_decrease' : [0, 0.0005, 0.001, 0.1, 0.3]}

# narrowed search after multiple iterations
register_config['xgbr'] = {'min_child_weight': [15,16],
                           'objective': ['reg:squarederror'],
                           'gamma': [0.8, 0.9],
                           'learning_rate': [0.01, 0.005],
                           'max_depth': [10, 12],
                           'subsample': [0.7, 0.8],
                           'n_estimators': [500, 600]}

register_config['lassor'] = {'alpha': [0.00005, 0.0001, 0.0005],
                             'selection': ['cyclic'],
                             'tol': [500]}

register_config['rfr'] = {'bootstrap': [True],
                          'n_estimators': [500, 600],
                          'max_features': ['sqrt'],
                          'min_samples_split': [5, 6],
                          'min_impurity_decrease': [0.0008, 0.001]}

with open('./mlp/params/register_params_search.py', 'w') as f:
    register_config.write(f)

##################################################
##get configurations for models on registered users
guest_config = ConfigParser()

# 1st iteration wide search
# guest_config['xgbr'] = {'min_child_weight' : [5,10,15],
#                     'objective' : ['reg:squarederror'],
#                     'gamma' : [0.1, 0.5, 0.8],
#                     'learning_rate' : [0.3, 0.1, 0.01, 0.001],
#                     'max_depth' : [5,10,20],
#                     'subsample' : [0.1,0.5,0.8],
#                     'n_estimators' : [100,500,1000]}

# guest_config['lassor'] = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                     'selection' : ['cyclic', 'random']}

# guest_config['rfr'] = {'bootstrap' : [True, False],
#                 'n_estimators' : [100,500,1000],
#                 'max_features' : ['sqrt', 'log2'],
#                 'min_samples_split' : [5, 10, 20],
#                 'min_impurity_decrease' : [0, 0.0005, 0.001, 0.1, 0.3]}

# narrowed search after multiple iterations
guest_config['xgbr'] = {'min_child_weight': [14, 15],
                        'objective': ['reg:squarederror'],
                        'gamma': [0.7, 0.8],
                        'learning_rate': [0.03, 0.02],
                        'max_depth': [8, 9],
                        'subsample': [0.7, 0.8],
                        'n_estimators': [500, 600]}

guest_config['lassor'] = {'alpha': [0.00005, 0.0001, 0.0005],
                          'selection': ['random'],
                          'tol': [500]}

guest_config['rfr'] = {'bootstrap': [True],
                       'n_estimators': [500, 600],
                       'max_features': ['sqrt'],
                       'min_samples_split': [4, 5],
                       'min_impurity_decrease': [0.0008, 0.001]}

with open('./mlp/params/guest_params_search.py', 'w') as f:
    guest_config.write(f)