# ML_model_service
A ML pipeline for basic XGBoost/LightGBM/simple NN model
  

## Set up  
pip install scikit-learn shap  

## arguments:
```
optional arguments:
  -h, --help             show this help message and exit
  --data_dir DATA_DIR    path to data
  --type TYPE            classification or regression
  --encoder ENCODER      categorical feature encoder, one of: label, onehot, target
  --algorithm ALGORITHM  which model you want to train(XGB or LGB or nn)
  --y_col Y_COL          column name of predict target
  --output_dir OUTPUT_DIR          path to output model
  --na_rule NA_RULE      (if you have domain rule)path to na rule(json)
  
```
## Training example:  
python train.py --y_col Survived --data_dir ./data/titanic.csv --output_dir ./models/titanic --exclude_col 'PassengerId'  
  
  
## REST API
**server:**  
python server.py --model_path ./models/titanic --shap_flag

**client:**  
python python_client.py --data_dir ./data/titanic_client_test.json 

## web service:  
python web_server.py  
  
