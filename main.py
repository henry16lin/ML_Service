import pandas as pd
import numpy as np
import os
import logging
import datetime
import time
import argparse
#import shap
import pickle
import json
import sys

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
    make_scorer, fbeta_score, recall_score, precision_score, mean_squared_error, mean_absolute_percentage_error)
import joblib

import seaborn as sns
from matplotlib import pyplot as plt

from preprocess import preprocess
from model import get_model


plt.style.use('seaborn')


cwd = os.getcwd()
abs_path = os.path.realpath(sys.argv[0])
normalLogger = logging.getLogger('normalLogger')


def drop_useful_col(df, exclude_col):
    # delet some col only for information(like id) but not useful in training 
    exclude_col = exclude_col.split(',')
    for c in df.columns:
        if str(df[c].dtypes)=='object':
            class_cnt = len(set(df[c]))
            if class_cnt>32:
                exclude_col.append(c)
    #exclude_col = ['PassengerId','id','Id']
    for c in exclude_col:
        if c in df.columns:
            df.drop([c],axis=1,inplace=True) 


def train(args):
    normalLogger.debug('loading data...')
    data = pd.read_csv(args.data_dir)
    
    normalLogger.debug('split data into train and test...')
    target, features = data[args.y_col], data.drop([args.y_col], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25,random_state=33)

    ##### output test_set.csv #####
    X_test[args.y_col] = y_test
    X_test.to_csv('val_set.csv',index=False)
    X_test.drop([args.y_col], axis=1, inplace=True)
    #####


    drop_useful_col(X_train, args.exclude_col)
    drop_useful_col(X_test, args.exclude_col)
    
    normalLogger.debug('X_train size:'+str(X_train.shape))
    normalLogger.debug('X_test size:'+str(X_test.shape))
    normalLogger.debug('y_train size:'+str(y_train.shape))
    normalLogger.debug('y_test size:'+str(y_test.shape))
    
    
    normalLogger.debug('create preprocess from training data...')
    preprocessor = preprocess( encoder=args.encoder, normalize=(args.algorithm=='nn') )
    
    if args.algorithm == 'nn':
        # note: target is for target encoder and nn to get output class count.
        #       if you don't use target encoder and algorithm is not nn, then target is not matter 
        X_train_encoder = preprocessor.fit_transform(X_train, auto_fill = True, target = y_train)
    else:
        X_train_encoder = preprocessor.fit_transform(X_train, auto_fill = False, target = y_train)
    

    # save preprocessor to pickle
    with open('./model_data/preprocessor.pkl', 'wb') as output:
        pickle.dump(preprocessor, output, pickle.HIGHEST_PROTOCOL)
    
    if args.type == "classification":
        scaler = sum(y_train!=1)/sum(y_train==1)
    else:
        scaler = 1 #in regression scaler is doesn't matter
    
    normalLogger.debug('initialize %s model...' %args.algorithm)
    model, param_grid = get_model(args.algorithm,
                                  type = args.type,
                                  scaler = scaler, 
                                  in_features = len(X_train.columns),
                                  num_classes = len(set(y_train)),
                                  mid_features = 256 )

    normalLogger.debug('getting model: ')
    normalLogger.debug(model)
    
    if args.algorithm =='nn': 
        
        from nn_factory import nn_factory
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        grid = nn_factory(model, device, X_train_encoder, y_train, batch_size=32)
        grid.fit(epoch=30) 
    
    else:    
        
        normalLogger.debug('Hyperparameter tuning...')
        grid_start = time.time()
        
        if args.type == "classification":
            scroer = make_scorer(fbeta_score , beta=3)
            #scroer = 'roc_auc'
        elif args.type == "regression":
            scroer = "neg_mean_absolute_error"
            # other metrics: https://scikit-learn.org/stable/modules/model_evaluation.html 

        if len(param_grid)>0:
            grid = GridSearchCV(estimator=model, cv=3, n_jobs=-1 , param_grid=param_grid, scoring=scroer, verbose=1)
            #grid = RandomizedSearchCV(estimator=model,cv=5, n_jobs=-1 , param_distributions=param_grid, scoring='f1_micro', n_iter=100)
        else:
            grid = model

        grid.fit(X_train_encoder, y_train)
        grid_end = time.time()-grid_start
        normalLogger.debug('finish grid search, it took %.5f min'%(grid_end/60))
        #print(grid.cv_results_)
    
        # save model for future inference
        normalLogger.debug('saving model to ./model_data')
        normalLogger.debug(grid.best_estimator_)
        joblib.dump(grid.best_estimator_, os.path.join(cwd,'model_data','model_%s.pkl'%args.algorithm))
    
        normalLogger.debug('saving feature importance')
        feature_importance(X_train_encoder,grid.best_estimator_)


    #see training performance
    normalLogger.debug('prediction on training set...')
    train_preds = grid.predict(X_train_encoder)
    
    if args.type =="classification":
        train_auc = roc_auc_score(y_train, train_preds)
        train_acc = accuracy_score(y_train, train_preds)
        train_recall = recall_score(y_train, train_preds)
        train_precision = precision_score(y_train, train_preds)

        normalLogger.debug('compute and save the confusion matrix...')
        train_conf = confusion_matrix(y_train, train_preds)

        # graph confusion table and save
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.set(font_scale=1.4)
        plt.figure()
        train_plot = sns.heatmap(train_conf, cmap=colormap, annot=True, cbar=False, fmt='d')
        train_fig = train_plot.get_figure()
        plt.title('train auc: %.3f, acc:%.3f, \nrecall:%s, precision:%s' %(train_auc, train_acc, str(round(train_recall,3)), str(round(train_precision,3)))) 
        train_fig.savefig("train_confusion.png")

    elif args.type =="regression":
        mape = mean_absolute_percentage_error(y_train, train_preds)
        rmse = mean_squared_error(y_train, train_preds, squared=False)
        corr = np.corrcoef(list(y_train), list(train_preds))[0][1]
        
        # graph scatterplot table and save
        plt.figure()
        plt.scatter(y_train, train_preds)
        plt.title("MAPE:{:.3f}, RMSE:{:.3f}, correlation:{:.3f}".format(mape,rmse,corr))
        plt.xlabel("true value")
        plt.ylabel("predict value")
        plt.savefig("train_scatter_plot.png")
        plt.close()


    ##### see testset performance #####
    normalLogger.debug('prediction on testing set...')
    normalLogger.debug('preprocess for testing set...')
    del preprocessor
    
    with open('./model_data/preprocessor.pkl', 'rb') as input:
        preprocessor_test = pickle.load(input)
    
    X_test_encoder  = preprocessor_test.transform(X_test)
    test_preds = grid.predict(X_test_encoder)
    
    if args.type =="classification":
        test_auc = roc_auc_score(y_test, test_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_recall = recall_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)
        test_conf = confusion_matrix(y_test, test_preds)

        # graph confusion table and save
        plt.figure()
        test_plot = sns.heatmap(test_conf, cmap=colormap, annot=True, cbar=False, fmt='d')
        test_fig = test_plot.get_figure()
        plt.title('test auc: %.3f,acc:%.3f, \nrecall:%s, precision:%s' %(test_auc,test_acc, str(round(test_recall,3)), str(round(test_precision,3) )) )
        test_fig.savefig("test_confusion.png")

    elif args.type =="regression":
        test_mape = mean_absolute_percentage_error(y_test, test_preds)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        test_corr = np.corrcoef(list(y_test), list(test_preds))[0][1]
        
        # graph scatterplot table and save
        plt.figure()
        plt.scatter(list(y_test), list(test_preds))
        plt.title("MAPE:{:.3f}, RMSE:{:.3f}, correlation:{:.3f}".format(test_mape,test_rmse,test_corr))
        plt.xlabel("true value")
        plt.ylabel("predict value")
        plt.savefig("test_scatter_plot.png")
        plt.close()

    

    
def feature_importance(X_train_encoder,model):
    importance_df = pd.DataFrame({'feature':list(X_train_encoder.columns),'importance':list(model.feature_importances_)})
    importance_df = importance_df.sort_values(by=['importance'],ascending=False)
    print(importance_df)
    plt.figure()
    import_plot = importance_df[:np.min([25,importance_df.shape[0]])].plot.bar(x='feature',y='importance',rot=90)
    tmp = import_plot.get_figure()
    tmp.savefig("feature_importance.png", bbox_inches="tight")
    
    
    

def inference(data, preprocessor, model):
    
    data_encoder= preprocessor.transform(data)
    #data_encoder.to_csv('data_encoder.csv',index=False)

    if args.algorithm != 'nn':

        normalLogger.debug('do inference...')
        inference_start = time.time()
        y_preds = model.predict(data_encoder)
        y_hat = np.expand_dims(y_preds,axis=0)
        try:
            preds_prob = model.predict_proba(data_encoder)
            print(preds_prob)
            pred_result = np.concatenate((y_hat.T, preds_prob), axis=1)
            print('predict probability:')
            print(preds_prob)

        except:
            preds_prob = [np.nan]*len(y_preds)
            preds_prob = np.expand_dims(preds_prob,axis=0)
            pred_result = np.concatenate((y_hat.T, preds_prob.T), axis=1)
            print('predict result:')
            print(y_hat)
        
        normalLogger.debug('finish inference, elapsed %.4fs'%(time.time()-inference_start))

        try:
            if len(data_encoder)==1:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data_encoder)
                #plt.figure()
                plt.switch_backend('agg')
                if len(shap_values) == 1:
                    local_explain_plot = shap.force_plot(explainer.expected_value,shap_values[0,:],data_encoder.iloc[0,:],show=False,matplotlib=True)
                else: #lgbm len(shap_values)==2
                    local_explain_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], data_encoder.iloc[0,:],show=False,matplotlib=True)
                #plt.title
                #plt.show()
                local_explain_plot.savefig("shap_importance.png")

            
        except:
            normalLogger.debug('fail to explain data by shap...')
            shap_values = []
        

    else:
        tensor_data = torch.from_numpy(np.array(data_encoder)).to(device)
        
        log_prob = F.log_softmax(model(tensor_data.float()))
        preds_prob = torch.exp(log_prob).data.cpu().numpy()
        print(preds_prob)

        y_preds = np.argmax(preds_prob,axis=1)
        y_hat = np.expand_dims(y_preds,axis=0)
        pred_result = np.concatenate((y_hat.T,preds_prob),axis=1)

        
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


def folder_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML model pipline for training and testing')
    parser.add_argument('--data_dir', default='./data/titanic.csv', help='path to data')
    parser.add_argument('--train', default=False, action="store_true",help='whether to train model')
    parser.add_argument('--type', default='classification',help='classification or regression')
    parser.add_argument('--encoder', default='label',help='categorical feature encoder, one of: label, onehot, target')
    parser.add_argument('--algorithm', default='XGB',help='which model you want to train(XGB or LGB or nn)')
    parser.add_argument('--y_col', default='Survived',help='column name of predict target')
    parser.add_argument('--model', default='./model_data/model_XGB.pkl', help='path to load model')
    parser.add_argument('--na_rule',  help='path to na rule(json)')
    parser.add_argument('--exclude_col', default='', help='col name you want to skip in training, string split with ","')
    args = parser.parse_args()

    folder_checker(os.path.join(cwd, 'model_data'))
    SetupLogger('normalLogger', "train.%Y-%m-%d.log")

    if args.train:
        normalLogger.debug('start to train...')
        train(args)
        normalLogger.debug('end train!\n')
    else:
        normalLogger.debug('start to inference...')
        
        # load preprocessor
        with open(os.path.join(cwd,'model_data','preprocessor.pkl'), 'rb') as pkl:
            preprocessor = pickle.load(pkl)
            
            
        # load the latest na rule and replace original
        try:
            with open(args.na_rule, 'r') as json_file:
                na_rule = json.load(json_file)
        except:
            na_rule = {}
        
        if na_rule:
            preprocessor.na_rule = na_rule
        print('NA fill rule (base on median or mode from training data): ')
        print(preprocessor.na_rule)
        
        # load model
        try:
            if os.path.basename(args.model).endswith('.pkl'): # sklearn model
                with open(args.model, 'rb') as f:
                     normalLogger.debug('load model .pkl file...')
                     model = joblib.load(f)
                 
            elif os.path.basename(args.model).endswith('.pt'): # pytorch model
                
                align_data = preprocessor.align_data
                train_target = preprocessor.target
                
                import torch
                import torch.nn.functional as F
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                normalLogger.debug('load model .pt file...')
                
                model, _ = get_model(args.algorithm,#scaler=scaler, 
                              type=args.type,
                              in_features=len(align_data.columns),
                              num_classes=len(set(train_target)),
                              mid_features=256 )
                
                state = torch.load(args.model)
                #state = torch.load('./checkpoint/epoch-5-val_loss0.037-val_acc0.972.pt')
                model.load_state_dict(state['state_dict'])
                print(model)
                model.eval()
                    
            normalLogger.debug('successfuly load model')
            
        except:
            normalLogger.debug('fail to load model...check model path or do training model')
            assert False, 'fail to load model'
        
        
        
        while True:
            # load data
            data_dir = input("input the data(csv) path:")
            
            data = pd.read_csv(data_dir)
            normalLogger.debug('successfuly load data')
            
            
            pred_result = inference(data,preprocessor,model)
            class_cnt = pred_result.shape[1]-1
            col_name = ['prediction'] + ['prob_'+str(i) for i in range(class_cnt)]
            
            pred_df = pd.DataFrame(pred_result,columns=col_name)
            data_with_pred = pd.concat([data, pred_df], axis=1)
            data_with_pred.to_csv('pred_result.csv',index=False)
            normalLogger.debug('end inference!\n')



