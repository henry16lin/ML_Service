# ML_model_service
A ML pipeline for basic XGBoost/LightGBM/simple NN model

## Set up  
pip install scikit-learn shap  
  
## Training:  
python main.py --train --y_col label --data_dir ./path/to/file.csv  
  
## Prediction: 
python main.py  
then it will ask you data path for prediction.  
  
## REST API
#### server:  
python server.py  
  
#### client:  
python python_client.py --data_dir ./your/json/file/path  
  
## web service:  
python web_server.py  
  
