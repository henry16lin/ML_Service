# -*- coding: utf-8 -*-
from flask import Flask, request, Response, abort
#import pandas as pd
import numpy as np
import os
import pickle
import logging
#from logging.handlers import TimedRotatingFileHandler
from logging.handlers import BaseRotatingHandler
import codecs

import datetime
import shap
import time
import json
import argparse
import joblib
from pandas import json_normalize

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



@app.route('/XGB', methods=['POST'])
def XGB():

    try:
        json_data = request.get_json()
        normalLogger.debug('receive json data...')

        normalLogger.debug('transform json to dataframe and do inference...')
        data = json_normalize(json_data)
        #print(data)
    except FileNotFoundError:
        normalLogger.debug('can not load json file, please check data...')
        abort(404)

    #try:
    inference_start = time.time()
    pred_prob, col_names, shap_values = inference(data)
    normalLogger.debug('finish inference, elapsed %.4fs (preprocess+shap+prediction)'%(time.time()-inference_start))
    
    #create result dict and transform to json format
    result_dict = {'pred_prob':pred_prob, 'feature_names':list(col_names), 'shap_values':shap_values}
    result_json = json.dumps(result_dict,cls=NumpyEncoder)
    
    #except:
    #    normalLogger.debug('fail to predict data...')
    #    abort(500)

    return Response(result_json,status=200, mimetype="application/json")




def inference(data):
    normalLogger.debug('preprocess data...')
    data_encoder = preprocessor.transform(data)
    #print(data_encoder)
    
    try:
        shap_start = time.time()
        explainer = shap.TreeExplainer(model)
        if len(data_encoder)==1:
            tmp_shap_values = explainer.shap_values(data_encoder)
            #print('base value:',explainer.expected_value)

            if len(tmp_shap_values)==2: #for lgbm
                shap_values = tmp_shap_values[1]
            else:
                shap_values = tmp_shap_values
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
                tmp_shap_values = explainer.shap_values(data_encoder[i:i+1])[0]
                if len(tmp_shap_values)== 2: #for lgbm
                     shap_values_single = tmp_shap_values[1]
                else:
                    shap_values_single = tmp_shap_values
                shap_values.append(shap_values_single)
        print(shap_values)

        normalLogger.debug('shap explainer elapsed %.4fs'%(time.time()-shap_start))

    except:
        normalLogger.debug('fail to explain data by shap...')
        shap_values = []
    
    normalLogger.debug('run model prediction...')

    try: # classification 
        pred_prob = np.round_(model.predict_proba(data_encoder),3)
        print('predict probability:')
        print(pred_prob)
        return [i[1] for i in pred_prob], data_encoder.columns, shap_values

    except: #regression
        y_preds = model.predict(data_encoder)
        return y_preds, data_encoder.columns, shap_values





def SetupLogger(loggerName, filename):
    path = os.path.join(cwd,'log')
    older_checker(path)

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



def MultiProcessLogger(loggerName, filename):
    logger = logging.getLogger(loggerName)
    
    path = os.path.join(cwd,'log')
    folder_checker(path)
    
    level = logging.DEBUG
    logfilename = os.path.join(path, filename)
    format = '%(asctime)s %(levelname)-8s %(message)s'
    hdlr = MultiProcessSafeDailyRotatingFileHandler(logfilename, encoding='utf-8')
    fmt = logging.Formatter(format)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    

# ref: https://my.oschina.net/lionets/blog/796438
class MultiProcessSafeDailyRotatingFileHandler(BaseRotatingHandler):
    """Similar with `logging.TimedRotatingFileHandler`, while this one is
    - Multi process safe
    - Rotate at midnight only
    - Utc not supported
    """
    def __init__(self, filename, encoding=None, delay=False, utc=False, **kwargs):
        self.utc = utc
        self.suffix = "%Y-%m-%d.log"
        self.baseFilename = filename
        self.currentFileName = self._compute_fn()
        BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)

    def shouldRollover(self, record):
        if self.currentFileName != self._compute_fn():
            return True
        return False

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.currentFileName = self._compute_fn()

    def _compute_fn(self):
        return self.baseFilename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        if self.encoding is None:
            stream = open(self.currentFileName, self.mode)
        else:
            stream = codecs.open(self.currentFileName, self.mode, self.encoding)
        # simulate file name structure of `logging.TimedRotatingFileHandler`
        if os.path.exists(self.baseFilename):
            try:
                os.remove(self.baseFilename)
            except OSError:
                pass
        try:
            os.symlink(self.currentFileName, self.baseFilename)
        except OSError:
            pass
        return stream





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='a ML model webAPI serving')
    parser.add_argument('--port', default='5566', help='port')

    args = parser.parse_args()

    
    #SetupLogger('normalLogger', "%Y-%m-%d %H:%M.log")
    MultiProcessLogger('normalLogger', 'server')

    
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
        debug = False,
        host = '0.0.0.0',
        port = args.port,
        threaded = False,
        processes=1
        #ssl_context = ('./ssl/XXX.crt', './ssl/XXX.key')
        )
    
    
    
    

