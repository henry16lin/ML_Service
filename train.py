import pandas as pd
import numpy as np
import os
import logging
import datetime
import time
import argparse
import pickle
import json
import sys

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
    make_scorer, fbeta_score, recall_score, precision_score, mean_squared_error, mean_absolute_percentage_error)
import joblib

import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('seaborn')

from preprocess import preprocess
from model import get_model
import util

cwd = os.getcwd()


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
    normalLogger.info('loading data...')
    data = pd.read_csv(args.data_dir)
    
    normalLogger.info('split data into train and test...')
    target, features = data[args.y_col], data.drop([args.y_col], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25,random_state=33)

    drop_useful_col(X_train, args.exclude_col)
    drop_useful_col(X_test, args.exclude_col)
    
    normalLogger.info('X_train size:'+str(X_train.shape))
    normalLogger.info('X_test size:'+str(X_test.shape))
    normalLogger.info('y_train size:'+str(y_train.shape))
    normalLogger.info('y_test size:'+str(y_test.shape))
    
    
    normalLogger.info('create preprocess from training data...')
    preprocessor = preprocess( encoder=args.encoder, normalize=(args.algorithm=='nn') )
    
    if args.algorithm == 'nn':
        # note: target is for target encoder and nn to get output class count.
        #       if you don't use target encoder and algorithm is not nn, then target is not matter 
        X_train_encoder = preprocessor.fit_transform(X_train, auto_fill = True, target = y_train)
    else:
        X_train_encoder = preprocessor.fit_transform(X_train, auto_fill = False, target = y_train)
    

    # save preprocessor to pickle
    with open(os.path.join(args.output_dir, 'preprocessor.pkl'), 'wb') as output:
        pickle.dump(preprocessor, output, pickle.HIGHEST_PROTOCOL)
    
    if args.type == "classification":
        scaler = int(sum(y_train != 1) / sum(y_train == 1))
        normalLogger.debug('the scaler is: %f' % scaler)
    else:
        scaler = 1 #in regression scaler is doesn't matter
    
    normalLogger.info('initialize %s model...' % args.algorithm)
    model, param_grid = get_model(args.algorithm,
                                  type=args.type,
                                  scaler=scaler, 
                                  in_features=len(X_train.columns),
                                  num_classes=len(set(y_train)),
                                  mid_features=256)

    normalLogger.debug('getting model: ')
    normalLogger.debug(model)
    
    if args.algorithm =='nn': 
        from nn_factory import nn_factory
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        normalLogger.info('use device: {} to train nn model'.format(device))
        final_model = nn_factory(model, device, X_train_encoder, y_train, 2, args.output_dir)
        final_model.fit(epoch=50) 
    
    else:    
        normalLogger.info('Hyperparameter tuning...')
        if args.type == "classification":
            scroer = make_scorer(fbeta_score, beta=1)  # 'roc_auc'
            #scroer = 'roc_auc'
        elif args.type == "regression":
            scroer = "neg_mean_absolute_error"
            # other metrics: https://scikit-learn.org/stable/modules/model_evaluation.html 

        train_start = time.time()
        if len(param_grid) > 0:
            grid = GridSearchCV(estimator=model, cv=3, n_jobs=-1, param_grid=param_grid, scoring=scroer, verbose=1)
            grid.fit(X_train_encoder, y_train)
            final_model = grid.best_estimator_
        else:
            final_model = model.fit(X_train_encoder, y_train)

        train_time = time.time() - train_start
        normalLogger.info('finish grid search and training, it took %.5f min' % (train_time / 60))

        # save model for future inference
        normalLogger.info(grid.best_estimator_)
        normalLogger.info('saving model to {}'.format(args.output_dir))
        joblib.dump(final_model, os.path.join(args.output_dir, 'model.pkl'))
    
        normalLogger.info('saving feature importance')
        feature_importance(X_train_encoder, final_model, args.output_dir)


    # see performance
    
    if args.type == 'classification': 
        normalLogger.info('prediction on training set...')
        train_preds_prob = final_model.predict_proba(X_train_encoder)[:, 1]
        train_preds = (train_preds_prob >= 0.5).astype(int)

        normalLogger.info('preprocess on testing set...')
        X_test_encoder = preprocessor.transform(X_test)
        normalLogger.info('prediction on testing set...')
        test_preds_prob = final_model.predict_proba(X_test_encoder)[:, 1]
        test_preds = (test_preds_prob >= 0.5).astype(int)

        util.get_confusion_table(y_train, train_preds, 'train', args.output_dir)
        util.get_prob_plot(y_train.values, train_preds_prob, 'train', args.output_dir)
        util.get_confusion_table(y_test, test_preds, 'test', args.output_dir)
        util.get_prob_plot(y_test.values, test_preds_prob, 'test', args.output_dir)

        pred_df = pd.DataFrame(train_preds_prob, columns=['prob'])
        pred_df = pd.DataFrame(test_preds_prob, columns=['prob'])

    elif args.type == 'regression':
        normalLogger.info('prediction on training set...')
        train_preds = final_model.predict(X_train_encoder)

        normalLogger.info('preprocess on testing set...')
        X_test_encoder  = preprocessor.transform(X_test)
        normalLogger.info('prediction on testing set...')
        test_preds = final_model.predict(X_test_encoder)

        util.get_scatter_eval(y_train, train_preds, 'train', args.output_dir)
        util.get_scatter_eval(y_test, test_preds, 'test', args.output_dir)

        pred_df = pd.DataFrame(train_preds, columns=['prob'])
        pred_df = pd.DataFrame(test_preds, columns=['prob'])

    # combine pred result back to train/test dataframe
    train_data_with_pred = pd.concat([X_train.reset_index(), pred_df], axis=1)
    train_data_with_pred['y'] = y_train.values
    test_data_with_pred = pd.concat([X_test.reset_index(), pred_df], axis=1)
    test_data_with_pred['y'] = y_test.values
    
    train_data_with_pred.to_csv(os.path.join(args.output_dir, 'train_with_pred.csv'), index=False)
    test_data_with_pred.to_csv(os.path.join(args.output_dir, 'test_with_pred.csv'), index=False)


def feature_importance(X_train_encoder, model, output_dir):
    importance_df = pd.DataFrame({'feature':list(X_train_encoder.columns),'importance':list(model.feature_importances_)})
    importance_df = importance_df.sort_values(by=['importance'],ascending=False)
    print(importance_df)
    plt.figure()
    import_plot = importance_df[:np.min([25,importance_df.shape[0]])].plot.bar(x='feature',y='importance',rot=90)
    tmp = import_plot.get_figure()
    tmp.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches="tight")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML model pipline for training and testing')
    parser.add_argument('--data_dir', default='./data/titanic.csv', help='path to data')
    parser.add_argument('--type', default='classification', help='classification or regression')
    parser.add_argument('--encoder', default='label', help='categorical feature encoder, one of: label, onehot, target')
    parser.add_argument('--algorithm', default='XGB', help='which model you want to train(XGB or LGB or nn)')
    parser.add_argument('--y_col', default='Survived', help='column name of predict target')
    parser.add_argument('--output_dir', default='./models', help='classification or regression')
    parser.add_argument('--na_rule', help='path to na rule(json)')
    parser.add_argument('--exclude_col', default='', help='col name you want to skip in training, string split with ","')
    args = parser.parse_args()

    util.folder_checker(os.path.join(cwd, 'logs'))
    util.folder_checker(os.path.join(cwd, args.output_dir))
    normalLogger = util.get_logger('normalLogger', 'debug', './logs/train.%Y-%m-%d.log')

    normalLogger.info('========== start to train ==========')
    train(args)
    normalLogger.info('========== end train! ==========\n')
