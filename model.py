import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from loss import CustomLoss


normalLogger = logging.getLogger('normalLogger')


def get_model(model_name, scaler=1, **kwargs):
    
    custom_loss = CustomLoss(alpha=scaler)
    param_grid={}

    # print(kwargs)
    # for k in kwargs.keys():
    #     print(k)
    #     print('%s = kwargs["%s"]' %(k,k))
    #     exec('%s = kwargs["%s"]' %(k,k))
    # print(in_features)
    
    if model_name == 'XGB':
        model =  XGBClassifier(n_jobs=-1, #n_estimators=50,
          random_state=100,
          colsample_bytree=0.8, subsample=0.8,importance_type='gain',
          scale_pos_weight=scaler
          #objective = custom_loss.focal_loss_boosting
          #max_delta_step=1
          )
        # onehot final tuning
        # param_grid = {'max_depth': [4],
        #               'min_child_weight':[4],
        #               'learning_rate':[0.1],
        #               'gamma':[0],
        #               'n_estimators':[100],
        #               'reg_alpha':[4],
        #               'reg_lambda':[7]
        #               }

        # tuning step suggestion: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        # # label final tuning
        param_grid = {'max_depth': [5],
                      'min_child_weight':[2],
                      'learning_rate':[0.1],
                      'gamma':[1],
                      'n_estimators':[100],
                      'reg_alpha':[3],
                      'reg_lambda':[10]
                      }

        #param_grid = {'n_estimators':[30]}

        # label with MONTH final tuning
        # param_grid = {'max_depth': [4],
        #               'min_child_weight':[1],
        #               'learning_rate':[0.05],
        #               'gamma':[1],
        #               'n_estimators':[200],
        #               'reg_alpha':[0],
        #               'reg_lambda':[4]
        #               }
        
    
    
    elif model_name == 'LGB':
        params_lgb = {
            #'scale_pos_weight': scaler,
            'objective' : custom_loss.focal_loss_boosting,
            'num_leaves': 60,
            #'reg_alpha': 2,  # L1, number of leaf
            #'reg_lambda': 4, # L2, leaf weight square
            'subsample':0.8,  #sample datas
            'colsample_bytree':0.8, #sample columns
            'objective':'binary',
            'class_weight':'balanced',
            'importance_type':'gain',
            'random_state':42,
            'n_jobs':-1,
            'silent':True
        }
        
        model = LGBMClassifier(**params_lgb)
            
        # param_grid = {
        #               'n_estimators': [50,75,100],
        #               'max_depth': [8,9,10],
        #               'learning_rate':[0.1],
        #               'min_child_samples': [20,30,40],
        #               'min_child_weight':[1.5, 2, 2.5, 3],
        #               'reg_alpha':[1,3,5,7],
        #               'reg_lambda':[2,4,6,8]
        #               }
        
        #  label with focal loss final tuning
        param_grid = {
                      'n_estimators': [50],
                      'max_depth': [9],
                      'learning_rate':[0.1],
                      'min_child_samples': [20],
                      'min_child_weight':[2.5],
                      'reg_alpha':[5],
                      'reg_lambda':[6]
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
                    nn.LeakyReLU(0.1, inplace=True),
                    #nn.Dropout(p=0.3),
                    
                    nn.Linear(mid_features, mid_features),
                    nn.BatchNorm1d(num_features = mid_features),
                    nn.LeakyReLU(0.1, inplace=True),
                    
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







