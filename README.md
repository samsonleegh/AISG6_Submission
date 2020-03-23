# Project Description
This project is to forecast the hourly number of active users for an e-scooter rental service in a city with features such as the date and various weather parameters. The project outcome includes a simple machine learning pipeline, the selected best models for 24-hr ahead forecast and the performance metrics.

# Models Selection
The current models selected for training and test in the pipeline are XGBoost Regressor, Lasso Regressor and RandomForest Regressor. As time trend element is present in user demand, the selection of ensembles / regression models are not appropriate unless the target has been stationarised or filtered with a time-series model prior. For our use case, a 24-hr difference during the pre-processing step is taken and stationarity has been tested for.

Lasso Regressor has been selected for its ability to regularise and tune out explanatory variables with a penalty. This is needed to reduce the presence of multicollinearity problem as variables such as weather and time are highly correlated. For eg. temperature/time/humidity...

Due to the stationarity of target variable, there is no longer linear relationships between target/explanatory variables. Lasso regressor might suffer from a bad fit and thus perform poorly - to be used as a baseline model. 

The presence of XGBoost and RandomForest Regressors do not require good linear relationships between explanatory and target variables. As the ensembles of decision trees only filter variables with respect to the stationary target they could ideally learn more effectively and shall be used.

# Models Evaluation
### Cross-validation scores
For the cross-validation results, XGBoost proved to be the best model for registered user predictions scoring a r-square of 0.83 and root-mean-sq error to target mean ratio at 0.37. Similarly XGBoost predicted guest user better as well, with r-square of 0.77 and root-mean-sq error to target mean ratio at 0.60. Lasso performs the worst as expected due to a lack of linear relationship between target/explanatory variables.
The poor performance of guest user prediction is hypothesised to be of two causes. 
1) Guest users demand are a lot more dynamic than registered users and are affected by wider environment factors such as public holidays, e-scooter promotional events, etc. 
2) There could be cannabilising effects as guest users switch to become registered users. The guest demand could thus convert into registered user demand with increased usage.
#
Further iterations to look into these hypothesis.
### Test scores
Final score based on the aggregated guest/registered user predictions from test set has a r-square of 0.87 and root-mean-sq error to target mean ratio at 0.32. The models out-of-sample test performs consistently (stationarity holds), based on the test results. The higher r-square when predicted targets are aggregated could be due to the fact that cannabilising effect is true. The over/under predicting from each model cancels the error from each other (possibly, for further investigation).

# Installation
The required packages to be installed can be found in the requirement.txt file.
```sh
pip install -r ./requirements.txt
```

# Project Pipeline & Usage
First create your own virtual environment.
```sh
pip install virtualenv
python -m virtualenv venv
source venv/Scripts/activate
```
Install the required dependencies.
```sh
pip install -r ./requirements.txt
```
Run the preprocessing file which ingests the raw data and outputs the cleaned datasets with engineered featueres as .npy files.
```sh
python ./mlp/preprocess.py
```
Configure the parameters you want to optimise for each model before running the command. The parameters can be changed directly on the file, after each iteration, without affecting other scripts. The parameters will then be pushed for gridsearch subsequently.
```sh
python ./mlp/config_param.py
```
Run the command to gridsearch for the best parameters for each model. Command outputs are the optimised models for prediction of registered and guest users.
```sh
python ./mlp/gridsearch_param.py
```
The default optimised models selected for validation and test are XGBoost Regressor, Lasso Regressor and RandomForest Regressor. Configure the file directly to select a subset of the models before running it. The chosen models will be pushed for evaluation.
```sh
python ./mlp/config_select_opt_models.py
```
The command will cross-validate the best registered/guest user prediction models respectively. The best models are then used to forecast the aggregated users on test set. The evaluation results, best models and test results are stored in the output file.
```sh
python ./mlp/model_evaluation.py
```

