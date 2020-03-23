# installing venv and activate
echo "setting up virtual env.."
python -m pip install virtualenv
python -m virtualenv venv
source venv/Scripts/activate
echo "virtual environment created"
# install dependencies
echo "installing dependencies..."
pip install -r ./requirements.txt
echo "dependencies installed..."
# run preprocessing
python ./mlp/preprocess.py
echo "Preprocessing completed..."
# update params
python ./mlp/config_param.py
echo "Params updated..."
# optimise models parameters
echo "Optimising models parameters..."
python ./mlp/gridsearch_param.py
echo "Models parameters optimised..."
# optimised models selection
echo "Selecting optimised models..."
python ./mlp/config_select_opt_models.py
echo "Optimised models selected..."
# Evaluate and test models
echo "Selecting optimised models..."
python ./mlp/model_evaluation.py
echo "Models evaluated and tested, please check files in output folder."