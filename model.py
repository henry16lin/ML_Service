import logging
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from loss import CustomLoss


normalLogger = logging.getLogger('normalLogger')


def get_model(model_name, type, scaler=1, **kwargs):
    # type = "classification" or "regression"
    #custom_loss = CustomLoss(alpha=scaler)
    param_grid={}
    
    if model_name == 'XGB':
        params_xgb = {
            'n_jobs': -1, 'random_state': 100, 'use_label_encoder':False,
            'colsample_bytree':0.9, 'subsample':0.9, 
            'importance_type':'gain', 'scale_pos_weight' :scaler
            }

        if type == "classification":
            model =  XGBClassifier(eval_metric='mlogloss', **params_xgb)

        elif type == "regression":
            model =  XGBRegressor(**params_xgb)
        else:
            raise ValueError("type must be 'classification' or 'regression'! ")

        # tuning step suggestion: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        param_grid = {'max_depth': [3,4],
                      #'min_child_weight':[2,3,4],
                      'learning_rate':[0.01, 0.05, 0.1],
                      'gamma':[0],
                      'n_estimators':[100,200],
                      'reg_alpha':[5],
                      'reg_lambda':[2]
                      }
    
    
    elif model_name == 'LGB':
        params_lgb = {
            #'scale_pos_weight': scaler,
            #'objective' : custom_loss.focal_loss_boosting,
            'num_leaves': 60,
            'subsample':0.8,  #sample datas
            'colsample_bytree':0.8, #sample columns
            'class_weight':'balanced',
            'importance_type':'gain',
            'random_state':42,
            'n_jobs':-1,
            'silent':True
        }
        
        if type == "classification":
            model = LGBMClassifier(**params_lgb)
        elif type == "regression":
            model = LGBMRegressor(**params_lgb)
        else:
            raise ValueError("type must be 'classification' or 'regression'! ")

        param_grid = {
                      'n_estimators': [100,200,300],
                      'max_depth': [3,5],
                      'learning_rate':[0.05, 0.1],
                      'min_child_samples': [10,20],
                      #'min_child_weight':[1.5, 2, 3],
                      'reg_alpha':[3,5],
                      'reg_lambda':[4,6]
                      }

        # #defaulet
        # model = LGBMClassifier()
        # param_grid={}

    

    elif model_name =='nn':
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, in_features, num_classes, mid_features):
                super(Net, self).__init__()

                self.classifier = nn.Sequential(
                    nn.Linear(in_features, mid_features),
                    nn.BatchNorm1d(num_features = mid_features),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    #nn.Dropout(p=0.3),
                    
                    nn.Linear(mid_features, mid_features),
                    nn.BatchNorm1d(num_features = mid_features),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    
                    nn.Linear(mid_features, num_classes)
                )
                
            def forward(self, x):
                x = self.classifier(x)
                return x

        # net = Net(30,256,2)
        # t = torch.randn(16, 30)
        # net(t)

        model = Net(kwargs['in_features'], kwargs['num_classes'], kwargs['mid_features'])
            
            
    return model, param_grid







