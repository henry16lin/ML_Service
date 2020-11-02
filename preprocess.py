import os
import logging
#import pickle
import json
from sklearn import preprocessing
import numpy as np
import pandas as pd


cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')


class preprocess():
    def __init__(self, na_rule_path=None):
        """
        na_rule_path: if there are some domain knowledge of missing value,
                      you can set it with json file in na_rule_path.
        le_dict: a label encoder dict for encoding categorical feature
        """
        
        self.le_dict = {}
        self.na_rule_path = na_rule_path
        if self.na_rule_path:
            try:
                with open(self.na_rule_path, 'r') as json_file:
                    self.na_rule = json.load(json_file)
            except:
                self.na_rule = {}
        else:
            self.na_rule_path = os.path.join(cwd,'model_data','na_rule.json')
            self.na_rule = {}
    
    
    
    def fit(self, data, auto_fill=True):
        
        normalLogger.debug('getting data dtypes...')
        self.dtype = data.dtypes.to_dict()
        
        for k in self.dtype.keys():
            self.dtype[k] = str(self.dtype[k])
        print(self.dtype)
        with open('./model_data/dtype.json', 'w') as outfile:
            json.dump(self.dtype, outfile)
        
        normalLogger.debug('fill na')
        data, self.na_rule = self.na_fill(data, self.na_rule, auto_fill=auto_fill)
        
        #normalLogger.debug('column type checking...')
        #data = self.mix_type_checker(data)

        normalLogger.debug('categorical feature encoder...')
        data_encoder, self.le_dict = self.cat_encoder(data)
        
        #save one data as reference for test data align onehot encoder
        self.align_data = data_encoder[0:1]
        
        
        
    def transform(self, data):
        
        normalLogger.debug('fill na...')
        data, na_rule = self.na_fill(data, self.na_rule, auto_fill=False) #not auto-fill when prediction

        
        normalLogger.debug('checking data dtypes...')
        for c in data.columns:
            if c in self.dtype.keys():
                if str(data[c].dtypes) != str( self.dtype[c] ):
                    normalLogger.debug('-- align %s type from %s to %s' %(c,str(data[c].dtypes),str( self.dtype[c] )) )
                    data[c] = data[c].astype(str(self.dtype[c]))
        

        #normalLogger.debug('column type checking...')
        #data = self.mix_type_checker(data)
        
        #data.to_csv('before_encoder.csv',index=False)################
        normalLogger.debug('categorical feature encoder...')
        data_encoder,le_dict = self.cat_encoder(data)
        
        ### see lebel encoder mapping ###
        #dict(zip(le.classes_, le.transform(le.classes_)))
        
        #data_encoder.to_csv('before_align.csv',index=False)###########
        
        normalLogger.debug('align feature with training data')
        _, data_encoder = self.align_data.align(data_encoder, join='left', axis=1, fill_value=0)
    
        return data_encoder, na_rule, le_dict
    
    
    def na_fill(self, data, rule, auto_fill):
        # if isn't training, set auto_fill=False
        df_c = data.copy()
        na_cnt = data.isnull().sum()
        null_col = list(na_cnt[na_cnt>0].index)
        fill_col = [c for c in null_col if c not in list(rule.keys())]
        
        if rule :
            for c in data.columns:
                if sum(data[c].isnull())>0:
                    if c in rule.keys():
                        df_c[c].fillna(rule[c],inplace=True,downcast=False)
                    else:
                        normalLogger.debug('-- no exists na rule to fill for column %s' %c)
        else:
            normalLogger.debug('-- no exists na rule to fill')
        
        if auto_fill:
            for col in fill_col:
                if data[col].dtypes=='object': #category fill by mode
                    fill_cat = data[col].value_counts().idxmax()
                    rule.update({col:fill_cat})
                    df_c[col].fillna(fill_cat,inplace=True,downcast=False)
        
                elif data[col].dtypes!='object':#numeric fill by median
                    fill_num = np.nanmedian(data[col])
        
                    rule.update({col:fill_num})
                    df_c[col].fillna(fill_num,inplace=True,downcast=False)
            
            #you might want to modify the NA rule, so output json file instead of adding into pipeline
            json_path = os.path.dirname(self.na_rule_path)
            if not os.path.exists(json_path):
                os.makedirs(json_path)
        
            with open(self.na_rule_path, 'w') as outfile:
                json.dump(rule, outfile)
    
        return df_c,rule
    
    
    def mix_type_checker(self,data):
    # it assumes that you have fill all missing value when call this function
    # for case like '1' and 1.0
    
        for c in data.columns:
            if data[c].dtypes == 'object':
                unique_type = set(list(map(type,data[c])))
                if len(unique_type) > 1:
                    print('%s has mix type data'%c)
                    for i in range(len( data[c])):
                        if type(data[c].iloc[i]) != str:
                            data[c].iloc[i] = str(int(data[c].iloc[i])) #replace with string
        
        return data
    
    
    

    def cat_encoder(self, df):
        # to-do: multi-type cat-encoder for specific column (like sklearn:ColumnTransformer)
        df, le_dict = self.oneHotEncode2(df, self.le_dict)    
        
        return df, le_dict
    
    
    
    def oneHotEncode(self, df, le_dict):
        if not le_dict:
            columnsToEncode = list(df.select_dtypes(include=['category','object']))
            train = True;
        else:
            columnsToEncode = le_dict.keys()   
            train = False;
    
        for feature in columnsToEncode:
            if train:
                le_dict[feature] = preprocessing.LabelEncoder()
            try:
                if train:
                    df[feature] = df[feature].astype(str)
                    df[feature] = le_dict[feature].fit_transform(df[feature])

                else:
                    if True:#df[feature].dtypes != 'object':
                        df[feature] = df[feature].astype(str)
                    
                    df[feature] = le_dict[feature].transform(df[feature])
                
                #print('finish fit transform')
                
                if len(df) == 1:
                    df = pd.concat([df,
                                pd.get_dummies(df[feature],drop_first=False).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
                else:
                    df = pd.concat([df,
                                    pd.get_dummies(df[feature],drop_first=True).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
                
                df = df.drop(feature, axis=1)

                normalLogger.debug('-- one-hot encoder for %s'%feature)

            except:
                normalLogger.debug('Error encoding '+feature+' there might be a new category in this feature.')
                normalLogger.debug(set(df[feature]))

                #df[feature]  = df[feature].convert_objects(convert_numeric='force')
                df[feature]  = df[feature].apply(pd.to_numeric, errors='coerce')
        return (df, le_dict)
        
        

    def oneHotEncode2(self, df, le_dict):
        if not le_dict:
            columnsToEncode = list(df.select_dtypes(include=['category','object']))
            train = True;
        else:
            columnsToEncode = le_dict.keys()   
            train = False;
    
        for feature in columnsToEncode:
            if train:
                le_dict[feature] = preprocessing.LabelEncoder()
            
            try:
                #df[feature] = df[feature].astype(str)
                
                if len(df) == 1:
                    df = pd.concat([df,
                                pd.get_dummies(df[feature],drop_first=False,dummy_na=False).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
                else:
                    df = pd.concat([df,
                                    pd.get_dummies(df[feature],drop_first=False,dummy_na=False).rename(columns=lambda x: feature + '_' + str(x))], axis=1)
                
                df = df.drop(feature, axis=1)

                normalLogger.debug('-- one-hot encoder for %s'%feature)

            except:
                normalLogger.debug('Error encoding '+feature+' there might be a new category in this feature.')
                normalLogger.debug(set(df[feature]))

                #df[feature]  = df[feature].convert_objects(convert_numeric='force')
                df[feature]  = df[feature].apply(pd.to_numeric, errors='coerce')
        return (df, le_dict)


    

