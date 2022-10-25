# -*- coding: utf-8 -*-
import os
import pickle
import logging
import datetime
import shap
import time
import json
import argparse
import joblib

import numpy as np
import pandas as pd
from flask import Flask, request, Response, abort

import util

app = Flask(__name__)
cwd = os.getcwd()



@app.route('/')
def home():
    return "<h1>This is a test page!</h1>"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        normalLogger.info('receive json data...')

        normalLogger.info('transform json to dataframe and do inference...')
        data = pd.json_normalize(json_data)
        # print(data)
    except FileNotFoundError:
        normalLogger.warning('can not load json file, please check data...')
        abort(404)

    inference_start = time.time()
    pred_prob, col_names, shap_values = inference(data, args.shap_flag)
    normalLogger.info('finish inference, elapsed %.4fs (preprocess+shap+prediction)' % (time.time() - inference_start))
    
    #create result dict and transform to json format
    result_dict = {'pred_prob':pred_prob, 'feature_names': list(col_names), 'shap_values': shap_values}
    result_json = json.dumps(result_dict, cls=util.NumpyEncoder)

    return Response(result_json,status=200, mimetype="application/json")


def inference(data, shap_flag):
    normalLogger.info('preprocess data...')
    data_encoder = preprocessor.transform(data)
    #print(data_encoder)
    
    if shap_flag:
        try:
            shap_start = time.time()
            explainer = shap.TreeExplainer(model)
            if len(data_encoder) == 1:
                tmp_shap_values = explainer.shap_values(data_encoder)
                # print('base value:', explainer.expected_value)

                if len(tmp_shap_values) == 2: #for lgbm 
                    shap_values = tmp_shap_values[1]
                else:
                    shap_values = tmp_shap_values
                # plt.figure()
                # plt.switch_backend('agg')
                # local_explain_plot = shap.force_plot(explainer.expected_value,shap_values[0,:],data_encoder.iloc[0,:],show=False,matplotlib=True)
                # plt.title
                # plt.show()
                # file_name = 'image.jpg'
                # local_explain_plot.savefig(os.path.join(save_path,file_name),bbox_inches="tight")
                
            else:
                shap_values = []
                for i in range(len(data_encoder)):
                    tmp_shap_values = explainer.shap_values(data_encoder[i:i + 1])[0]
                    if len(tmp_shap_values) == 2: # for lgbm
                        shap_values_single = tmp_shap_values[1]
                    else:
                        shap_values_single = tmp_shap_values
                    shap_values.append(shap_values_single)

            normalLogger.info('shap explainer elapsed %.4fs' % (time.time() - shap_start))

        except:
            normalLogger.warning('fail to explain data by shap...')
            shap_values = []
    
    else:
        shap_values = []
    
    normalLogger.info('run model prediction...')
    try:  # classification 
        pred_prob = np.round_(model.predict_proba(data_encoder), 3)
        print('predict probability:')
        print(pred_prob)
        return [i[1] for i in pred_prob], data_encoder.columns, shap_values

    except:  # regression
        y_preds = model.predict(data_encoder)
        return y_preds, data_encoder.columns, shap_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='a ML model webAPI serving')
    parser.add_argument('--port', default='5566', help='port')
    parser.add_argument('--model_path', default='./models', help='port')
    parser.add_argument('--shap_flag', default=False, action="store_true", help='whether to use shap to explain the prediction')
    args = parser.parse_args()

    util.folder_checker(os.path.join(cwd, 'logs'))
    normalLogger = util.MultiProcessLogger('normalLogger',  './logs/server.%Y-%m-%d.log')

    normalLogger.info('load model...')
    with open(os.path.join(args.model_path, 'model.pkl'), 'rb') as f:
        model = joblib.load(f)
    
    normalLogger.info('load preprocess...')
    with open(os.path.join(args.model_path, 'preprocessor.pkl'), 'rb') as pkl:
        preprocessor = pickle.load(pkl)

    app.run(
        debug = False,
        host = '0.0.0.0',
        port = args.port,
        threaded = False,
        processes=1
        #ssl_context = ('./ssl/XXX.crt', './ssl/XXX.key')
        )
    
    
    
    

