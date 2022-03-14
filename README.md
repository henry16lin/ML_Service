# ML_model_service
A ML pipeline for basic XGBoost/LightGBM/simple NN model

## Set up  
pip install scikit-learn shap  

## arguments:
```
optional arguments:
  -h, --help             show this help message and exit
  --data_dir DATA_DIR    path to data
  --train                whether to train model
  --type TYPE            classification or regression
  --encoder ENCODER      categorical feature encoder, one of: label, onehot, target
  --algorithm ALGORITHM  which model you want to train(XGB or LGB or nn)
  --y_col Y_COL          column name of predict target
  --model MODEL          path to load model(for prediction)
  --na_rule NA_RULE      (if you have domain rule)path to na rule(json)
  
```
## Training example:  
python main.py --train --y_col Survived --data_dir ./data/titanic.csv --exclude_col 'PassengerId'  
  
## Prediction example: 
python main.py  
then it will ask you data path(.csv) for prediction.  

## REST API
**server:**  
python server.py

**client:**  
python python_client.py --data_dir ./data/titanic_client_test.csv 

## web service:  
python web_server.py  
  
