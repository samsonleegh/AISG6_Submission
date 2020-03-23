"""
Select optimised models for evaluation and test
"""
from configparser import ConfigParser

opt_models_select = ConfigParser()

opt_models_select['model_nm_dict'] = {'xgbr': ['XGBoostRegressor'],
                                      'rfr': ['RandomForestRegressor'],
                                      'lassor': ['LassoRegressor']}

with open('./mlp/optimised_models/opt_models_select.py', 'w') as f:
    opt_models_select.write(f)