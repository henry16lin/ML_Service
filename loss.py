import numpy as np


# y_hat on XGBClassifier is post-sigmoid response

class CustomLoss:
    def __init__(self,alpha):
        self.alpha = alpha

    def alpha_balance_CE(self, y_true, y_pred):
        a = self.alpha
        y = y_true
        p = y_pred
        
        grad = -( a*y/p - (1-y)/(1-p) )
        hess = -( -a*y/(p**2) - (1-y)/((1-p)**2) )
        
        return grad, hess

    #https://blog.csdn.net/qq_32103261/article/details/108219332
    def focal_loss_boosting(self, y_true,y_pred):
        a = self.alpha
        g = 2
        y = y_true
        p=y_pred
        
        grad=p*(1 - p)*(g*a*y*(1 - p)**g*np.log(p)/(1 - p) - g*p**g*(1 - y)*np.log(1 - p)/p - a*y*(1 - p)**g/p + p**g*(1 - y)/(1 - p))
        hess=p*(1 - p)*(p*(1 - p)*(-g**2*a*y*(1 - p)**g*np.log(p)/(1 - p)**2 - g**2*p**g*(1 - y)*np.log(1 - p)/p**2 + g*a*y*(1 - p)**g*np.log(p)/(1 - p)**2 + 2*g*a*y*(1 - p)**g/(p*(1 - p)) + 2*g*p**g*(1 - y)/(p*(1 - p)) + g*p**g*(1 - y)*np.log(1 - p)/p**2 + a*y*(1 - p)**g/p**2 + p**g*(1 - y)/(1 - p)**2) - p*(g*a*y*(1 - p)**g*np.log(p)/(1 - p) - g*p**g*(1 - y)*np.log(1 - p)/p - a*y*(1 - p)**g/p + p**g*(1 - y)/(1 - p)) + (1 - p)*(g*a*y*(1 - p)**g*np.log(p)/(1 - p) - g*p**g*(1 - y)*np.log(1 - p)/p - a*y*(1 - p)**a/p + p**g*(1 - y)/(1 - p)))
        
        return grad, hess


