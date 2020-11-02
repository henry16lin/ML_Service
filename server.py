# -*- coding: utf-8 -*-
from flask import Flask, request, Response
import pandas as pd
import numpy as np
import os
import pickle
import logging
import datetime
import shap
import time
import json
from sklearn.externals import joblib
from pandas.io.json import json_normalize

app = Flask(__name__)

cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/')
def home():
    return "<h1>This is a test page!</h1>"



@app.route('/SEPSIS_XGB', methods=['POST'])
def SEPSIS_XGB():
    json_data = request.get_json()
    normalLogger.debug('receive json data...')
    print(json_data)

    normalLogger.debug('transform json to dataframe and do inference...')
    data = json_normalize(json_data)
    print(data)

    inference_start = time.time()
    pred_prob, col_names, shap_values = inference(data)
    normalLogger.debug('finish inference, elapsed %.4fs (preprocess+shap+prediction)'%(time.time()-inference_start))
    
    #create result dict and transform to json format
    result_dict = {'pred_prob':pred_prob, 'feature_names':list(col_names), 'shap_values':shap_values}
    result_json = json.dumps(result_dict,cls=NumpyEncoder)
    
    return Response(result_json,status=200, mimetype="application/json")




def inference(data):
    normalLogger.debug('preprocess data...')
    data_encoder, _, _ = preprocessor.transform(data)
    
    shap_start = time.time()
    explainer = shap.TreeExplainer(model)
    if len(data_encoder)==1:
        
        shap_values = explainer.shap_values(data_encoder)
        
        #plt.figure()
        #plt.switch_backend('agg')
        #local_explain_plot = shap.force_plot(explainer.expected_value,shap_values[0,:],data_encoder.iloc[0,:],show=False,matplotlib=True)
        #plt.title
        #plt.show()
        #file_name = 'image.jpg'
        #local_explain_plot.savefig(os.path.join(save_path,file_name),bbox_inches="tight")
        
    else:
        shap_values=[]
        for i in range(len(data_encoder)):
            shap_values.append(explainer.shap_values(data_encoder[i:i+1])[0] )
    
    normalLogger.debug('shap explainer elapsed %.4fs'%(time.time()-shap_start))
    
    #shap_values=[]
    pred_prob = np.round_(model.predict_proba(data_encoder),3)
    print(pred_prob)
    #y_hat = np.expand_dims(y_preds,axis=0)
    #pred_result = np.concatenate((y_hat.T,pred_prob),axis=1)
    

    return [i[1] for i in pred_prob], data_encoder.columns, shap_values





def SetupLogger(loggerName, filename):
    path = os.path.join(cwd,'log')
    if not os.path.exists(path):
        os.makedirs(path)

    logger = logging.getLogger(loggerName)

    logfilename = datetime.datetime.now().strftime(filename)
    logfilename = os.path.join(path, logfilename)

    logformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(logfilename, 'a', 'utf-8')
    fileHandler.setFormatter(logformatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logformatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)



def folder_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)




if __name__ == '__main__':
    
    
    SetupLogger('normalLogger', "%Y-%m-%d.log")
    
    #load model
    with open('./model_data/grid.pkl','rb') as f:
        model = joblib.load(f)
        
    
    # load preprocessor
    with open('./model_data/preprocessor.pkl', 'rb') as pkl:
        preprocessor = pickle.load(pkl)    
    
    
    # load the latest na rule and replace original
    try:
        with open('./model_data/na_rule.json', 'r') as json_file:
            na_rule = json.load(json_file)
    except:
        na_rule = {}
    
    
    if na_rule:
        preprocessor.na_rule = na_rule
    
    
    app.run(
        debug = True,
        host = '0.0.0.0',
        port = 5566,
        threaded = False,
        processes=4
        #ssl_context = ('./ssl/XXX.crt', './ssl/XXX.key')
        )
    
    
    
    

