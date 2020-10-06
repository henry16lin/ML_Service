# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file,abort
import pandas as pd
import numpy as np
import uuid
import os
import pickle
import logging
import datetime
import shap
from matplotlib import pyplot as plt
import time
import json
from sklearn.externals import joblib
import threading
import shutil
from glob import glob


app = Flask(__name__)

cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')




@app.route('/')
def home():
    return "<h1>This is a test page!</h1>"

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and request.form['submit_button'] == 'submit':
        df = pd.read_csv(request.files.get('file'))
        
        uid = str(uuid.uuid1())
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = '%s_%s.csv'%(time_str,uid)
        df.to_csv('./pred_file/%s'%csv_name, index=False)
        
        #return render_template('result.html', shape=df.shape)
        #return redirect(url_for('result', token=uuid))
        return render_template('loading.html',
                               loading_img='./static/loading.gif',
                               uid = uid, time_str = time_str)
    
    
    
    return render_template('upload.html',na_rule=na_rule)
 


@app.route('/result/<time_str>/<uid>')
def result(time_str,uid):
    data = pd.read_csv('./pred_file/%s_%s.csv'%(time_str,uid))
    
    save_path = os.path.join(cwd,'static',uid)
    
    pred_prob = inference(data,save_path)
    prob_str = 'probability: '+ str(pred_prob)
    #image_path = os.path.join(cwd,'static',uid,'image.jpg')
    
    
    '''
    hist_jpg = os.path.join('static',token,'hist.jpg')
    box_jpg = os.path.join('static',token,'box.jpg')
    prob_jpg = os.path.join('static',token,'prob.jpg')
    trend_jpg = os.path.join('static',token,'trend.jpg')
    '''
    
    return render_template("result.html", 
                           prob = prob_str,
                           image_path='../../../../../../static/%s/image.jpg'%uid)
                           #messages=request.args.get('messages'))


@app.route('/get-csv')
def get_csv():
    try:
        filepath = os.path.join(cwd,'static','upload_format.csv')
        print(filepath)
        return send_file(filepath, as_attachment=True,cache_timeout=0)
    except FileNotFoundError:
        abort(404)



def inference(data,save_path):
    
    
    data_encoder, _, _ = preprocessor.transform(data)
    
    
    folder_checker(save_path)
    if len(data_encoder)==1:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_encoder)
        
        #plt.figure()
        local_explain_plot = shap.force_plot(explainer.expected_value,shap_values[0,:],data_encoder.iloc[0,:],show=False,matplotlib=True)
        plt.title
        #plt.show()
        file_name = 'image.jpg'
        local_explain_plot.savefig(os.path.join(save_path,file_name),bbox_inches="tight")
        
    else:
        y_preds = model.predict(data_encoder)
        make_pie(y_preds,save_path)
    
    
    normalLogger.debug('do inference...')
    inference_start = time.time()
    pred_prob = np.round_(model.predict_proba(data_encoder),3)
    
    #y_hat = np.expand_dims(y_preds,axis=0)
    #pred_result = np.concatenate((y_hat.T,pred_prob),axis=1)
    
    normalLogger.debug('finish inference, elapsed %.4fs'%(time.time()-inference_start))
    
    
    #return pred_result
    return pred_prob



def make_pie(y_preds,save_path):
    data_cnt = len(y_preds)
    pass_cnt = sum(y_preds==1)
    fail_cnt = data_cnt-pass_cnt

    plt.figure()
    labels = 'Sepsis','no Sepsis'
    plt.pie([pass_cnt,fail_cnt],labels=labels,autopct=make_autopct([pass_cnt,fail_cnt]))
    plt.title('total data count: %d'%data_cnt)
    plt.savefig (os.path.join(save_path,'image.jpg'))


def make_autopct(values):#for pi chart symbol
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct




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
    #path = os.path.join(cwd,'model_data')
    if not os.path.exists(path):
        os.makedirs(path)



def old_pred_file_cleaner():
    while True:
        for f in glob(os.path.join(cwd,'pred_file',"*.csv")):
            mtime = os.path.getmtime(f)
            mtime_ = datetime.datetime.fromtimestamp(mtime)
            now_time = datetime.datetime.now()
            
            if mtime_< (now_time- datetime.timedelta(minutes=10)):
                normalLogger.debug('delete old pred file:%s'%f)
                os.remove(f)
        time.sleep(600)
    

def old_token_cleaner():
    while True:
        for folder in glob(os.path.join(cwd,'static','*/')):
            mtime = os.path.getmtime(folder)
            mtime_ = datetime.datetime.fromtimestamp(mtime)
            now_time = datetime.datetime.now()
            if mtime_< (now_time- datetime.timedelta(minutes=10)):
                normalLogger.debug('delete old graph:%s'%folder)
                shutil.rmtree(folder)
        time.sleep(600)




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
    
    
    
    # sub-tread check old file
    cleaner1 = threading.Thread(target = old_pred_file_cleaner, daemon=True)
    cleaner2 = threading.Thread(target = old_token_cleaner,daemon=True)

    cleaner1.start()
    cleaner2.start()
    
    folder_checker(os.path.join(cwd,'pred_file'))
    
    app.run(
        debug = True,
        host = '0.0.0.0',
        port = 5566,
        threaded = False,
        ssl_context = ('./ssl/XXX.crt', './ssl/XXX.key')
        )
    
    



