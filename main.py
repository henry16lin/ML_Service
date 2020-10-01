import pandas as pd
import numpy as np
import os
import logging
import datetime
import time
import argparse
import shap
import pickle

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt

from preprocess import preprocess

plt.style.use('ggplot')



cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')



def train(args):
    normalLogger.debug('loading data...')
    data = pd.read_csv(args.data_dir)
    

    normalLogger.debug('split data into train and test...')
    target, features = data[args.y_col], data.drop([args.y_col], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=100)

    normalLogger.debug('X_train size:'+str(X_train.shape))
    normalLogger.debug('X_test size:'+str(X_test.shape))
    normalLogger.debug('y_train size:'+str(y_train.shape))
    normalLogger.debug('y_test size:'+str(y_test.shape))
    
    
    normalLogger.debug('create preprocess from training data...')
    preprocessor = preprocess()
    preprocessor.fit(X_train)
    X_train_encoder, na_rule, le_dict = preprocessor.transform(X_train)
    
    # save preprocessor to pickle
    with open('./model_data/preprocessor.pkl', 'wb') as output:
        pickle.dump(preprocessor, output, pickle.HIGHEST_PROTOCOL)
    
    
    scaler = sum(y_train!=1)/sum(y_train==1)
    #restrict the max scale time
    if sum(y_train!=1)/sum(y_train==1)>100:
        scaler = 100 #np.floor(np.sqrt(sum(y_train!=1)/sum(y_train==1)))
    
    model =  XGBClassifier(n_estimators=100, n_thread=4,
                                                random_state=100,reg_alpha=1,reg_lambda=2,
                                                colsample_bytree=0.8,subsample=0.8,
                                                importance_type='gain',
                                                scale_pos_weight=scaler)
    

    normalLogger.debug('Hyperparameter tuning with grid search...')
    scorer = make_scorer(roc_auc_score, greater_is_better=True)

    param_grid = {'xgb__max_depth': [3,4],
                  'xgb__min_child_weight':[0.5,1],
                  'xgb__learning_rate':[0.05,0.1]}

    grid_start = time.time()
    grid = GridSearchCV(estimator=model,cv=3, param_grid=param_grid, scoring=scorer)
    grid.fit(X_train_encoder, y_train)
    grid_end = time.time()-grid_start
    normalLogger.debug('finish grid search, it took %.5f min'%(grid_end/60))

    # save model for future inference
    normalLogger.debug('saving model to ./model_data')
    normalLogger.debug(grid.best_estimator_)
    joblib.dump(grid.best_estimator_, os.path.join(cwd,'model_data','grid.pkl'))

    normalLogger.debug('saving feature importance')
    feature_importance(X_train_encoder,grid.best_estimator_)


    #see training performance
    normalLogger.debug('prediction on training set...')
    train_preds = grid.predict(X_train_encoder)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_preds, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)

    
    normalLogger.debug('compute and save the confusion matrix...')
    train_conf = confusion_matrix(y_train, train_preds)
    
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure()
    train_plot = sns.heatmap(train_conf,cmap=colormap,annot=True,cbar=False,fmt='d')
    train_fig = train_plot.get_figure()
    plt.title('train auc: %.3f' %train_auc) 
    train_fig.savefig("train_confusion.png")
    
    ##### see testset performance #####
    normalLogger.debug('prediction on testing set...')
    normalLogger.debug('preprocess for testing set...')
    del preprocessor
    
    with open('./model_data/preprocessor.pkl', 'rb') as input:
        preprocessor_test = pickle.load(input)
    
    X_test_encoder, _, _ = preprocessor_test.transform(X_test)
    #X_test_encoder, na_rule, le_dict = preprocessor(X_test,na_rule=na_rule,le_dict=le_dict,train_ind=False)
    
    test_preds = grid.predict(X_test_encoder)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_preds, pos_label=1)
    test_auc = metrics.auc(fpr,tpr)
    
    test_conf = confusion_matrix(y_test, test_preds)
    
    plt.figure()
    test_plot = sns.heatmap(test_conf,cmap=colormap,annot=True,cbar=False,fmt='d')
    test_fig = test_plot.get_figure()
    plt.title('test auc: %.3f' %test_auc) 
    test_fig.savefig("test_confusion.png")

    
    
def feature_importance(X_train_encoder,model):
    importance_df = pd.DataFrame({'feature':list(X_train_encoder.columns),'importance':list(model.feature_importances_)})
    importance_df = importance_df.sort_values(by=['importance'],ascending=False)
    print(importance_df.head())
    plt.figure()
    import_plot = importance_df[:np.min([20,importance_df.shape[0]])].plot.bar(x='feature',y='importance',rot=90)
    tmp = import_plot.get_figure()
    tmp.savefig("feature_importance.png",bbox_inches="tight")
    
    
    

def inference(data, preprocessor, model):
    data_encoder, na_rule, le_dict = preprocessor.transform(data)
    
    if len(data_encoder)==1:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_encoder)
        
        #plt.figure()
        local_explain_plot = shap.force_plot(explainer.expected_value,shap_values[0,:],data_encoder.iloc[0,:],show=False,matplotlib=True)
        plt.title
        plt.show()
        local_explain_plot.savefig("shap_importance.png",bbox_inches="tight")
        
    
    
    normalLogger.debug('do inference...')
    inference_start = time.time()
    y_preds = model.predict(data_encoder)
    preds_prob = model.predict_proba(data_encoder)
    
    y_hat = np.expand_dims(y_preds,axis=0)
    pred_result = np.concatenate((y_hat.T,preds_prob),axis=1)
    
    normalLogger.debug('finish inference, elapsed %.4fs'%(time.time()-inference_start))

    return pred_result




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


def folder_checker():
    path = os.path.join(cwd,'model_data')
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basic_XGB_builder')
    parser.add_argument('--data_dir', default='./pure_data.csv', help='path to data')
    parser.add_argument('--train', default=False, action="store_true",help='whether to train model(XGBoost)')
    parser.add_argument('--y_col', default='label',help='column name of predict target')
    parser.add_argument('--model', default='./model_data/grid.pkl', help='path to load model')
    parser.add_argument('--na_rule', default='./model_data/na_rule.json', help='path to na rule(json)')
    args = parser.parse_args()

    folder_checker()
    SetupLogger('normalLogger', "%Y-%m-%d.log")

    if args.train:
        normalLogger.debug('start to train...')
        train(args)
        normalLogger.debug('end train!\n')
    else:
        normalLogger.debug('start to inference...')
        
        #load model
        try:
            with open(args.model, 'rb') as f:
                model = joblib.load(f)
            normalLogger.debug('successfuly load model')
        except:
            normalLogger.debug('fail to load model...check model path or do training model')
            assert False, 'fail to load model'
        
        # load preprocessor
        with open('./model_data/preprocessor.pkl', 'rb') as pkl:
            preprocessor = pickle.load(pkl)
        
        
        while True:
            # load data
            data_dir = input("input the data(csv) path:")
            print(data_dir)
            print(type(data_dir))
            data = pd.read_csv(data_dir)
            normalLogger.debug('successfuly load data')
            
            
            pred_result = inference(data,preprocessor,model)
            class_cnt = pred_result.shape[1]-1
            col_name = ['prediction'] + ['prob_'+str(i) for i in range(class_cnt)]
            
            pred_df = pd.DataFrame(pred_result,columns=col_name)
            pred_df.to_csv('pred_result.csv',index=False)
            normalLogger.debug('end inference!\n')



