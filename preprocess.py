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
    def __init__(self,encoder='onehot', na_rule_path=None, normalize=False):
        """
        - encoder: str, which kind of categorical encoder to use (one of 'onehot', 'label' , 'target' )
        - na_rule_path: None or str, if there are some domain knowledge of missing value,
                        you can set it with json file in na_rule_path.
        - normalize: bool, whether to normalize data to [0,1] (i.e. min_max_scaler) after categorical encoder
        """
        
        self.le_dict = {}
        self.na_rule_path = na_rule_path
        self.normalize = normalize
        self.encoder = encoder

        if self.na_rule_path:
            try:
                with open(self.na_rule_path, 'r') as json_file:
                    self.na_rule = json.load(json_file)
            except:
                self.na_rule = {}
        else:
            self.na_rule_path = os.path.join(cwd,'model_data','na_rule.json')
            self.na_rule = {}
    
    
    
    def fit_transform(self, data, auto_fill=True, target=None):
        
        self.target =target

        normalLogger.debug('getting data dtypes...')
        self.dtype = data.dtypes.to_dict()

        for k in self.dtype.keys():
            self.dtype[k] = str(self.dtype[k])
        #print(self.dtype)
        with open('./model_data/dtype.json', 'w') as outfile:
            json.dump(self.dtype, outfile)
        
        normalLogger.debug('fill na')
        data, self.na_rule = self.na_fill(data, self.na_rule, auto_fill=auto_fill)
        
        #normalLogger.debug('column type checking...')
        #data = self.mix_type_checker(data)

        normalLogger.debug('categorical feature encoder...')
        data_encoder = self.cat_encoder(data, is_train=True, target = target)
        

        if self.normalize:
            normalLogger.debug('normalize data...')
            data_encoder = self._normalize(data_encoder, is_train=True)


        return data_encoder
        
        
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
        
        normalLogger.debug('categorical feature encoder...')
        data_encoder = self.cat_encoder(data, is_train=False)
        
        ### see lebel encoder mapping ###
        # le = self.le_dict['some_column']
        #dict(zip(le.classes_, le.transform(le.classes_)))

        if self.normalize:
            normalLogger.debug('normalize data...')
            data_encoder = self._normalize(data_encoder, is_train=False)

        return data_encoder
    
    

    def _normalize(self, df, is_train):
        tmp_df = df.copy()
        if is_train:
            self.normalizer = preprocessing.MinMaxScaler()
            tmp_np = self.normalizer.fit_transform(tmp_df)

        else:
            tmp_np = self.normalizer.transform(tmp_df)

        return_df = pd.DataFrame(tmp_np, columns=list(df.columns))

        return return_df




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
    
    
    

    def cat_encoder(self, df, is_train, target=None):
        # to-do: multi-type cat-encoder for specific column (like sklearn:ColumnTransformer)
        # it can be user specific or automatically use cardinality to detemine which column use onehot
        
        normalLogger.debug('-- implement %s encoder' %self.encoder)

        if self.encoder == 'label':
            data_encoder = self.label_encoder(df, is_train)

        elif self.encoder == 'target':
            data_encoder = self.target_encoder(df, is_train, target)

        elif self.encoder == 'onehot':
            data_encoder = self.oneHotEncode(df, is_train)

        return data_encoder
    
    

    def label_encoder(self, df, is_train):
        categorical_features = list(df.select_dtypes(include=['category','object']))
        tmp_df = df.copy()
        if is_train:
            for c in categorical_features:
                self.le_dict[c] = preprocessing.LabelEncoder()
                normalLogger.debug('-- label encoder for %s' %c )
                tmp_df[c] = self.le_dict[c].fit_transform(tmp_df[c].astype(str))
                ''' 
                Here, treat na as a category. if there is some na rule in domain knowledge, than na_fill will be process before
                if don't want to remain na in column, can consider below page:
                ref: https://stackoverflow.com/questions/36808434/label-encoder-encoding-missing-values
                '''

                self.align_data = tmp_df[0:1] 
        else:
            for c in categorical_features:
                if c not in list(self.le_dict.keys()): #if new column, then skip it
                    continue
                le = self.le_dict[c]
                #normalLogger.debug('-- label encoder for %s' %c )
                mapping = dict(zip(le.classes_, le.transform(le.classes_))) # get mapping dict from label encoder
                tmp_df[c] = tmp_df[c].apply(lambda x: mapping.get(x, -1)) 
                # if there are new category, then it will not map to any key and give it -1
                # ref: https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values

            _, tmp_df = self.align_data.align(tmp_df, join='left', axis=1, fill_value= -1)

        return tmp_df



    def oneHotEncode(self, df, is_train):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        normalLogger.debug('-- one-hot encoder for %s'%str(columnsToEncode))

        if len(df) ==1:
            data_encoder = pd.get_dummies(df, columns=columnsToEncode,drop_first=False,dummy_na=False)
        else:
            data_encoder = pd.get_dummies(df, columns=columnsToEncode,drop_first=False,dummy_na=False) # here can consider drop_first to be True

        if is_train:
            self.align_data = data_encoder[0:1] #save one data as reference for test data align onehot encoder
        else:
            normalLogger.debug('align feature with training data')
            _, data_encoder = self.align_data.align(data_encoder, join='left', axis=1, fill_value=0)

        return data_encoder


    def target_encoder(self, df, is_train, target):
        from category_encoders import TargetEncoder
        #from category_encoders import CatBoostEncoder
        # https://contrib.scikit-learn.org/category_encoders/targetencoder.html#

        categorical_features = list(df.select_dtypes(include=['category','object']))
        tmp_df = df.copy()
        if is_train:
            for c in categorical_features:
                self.le_dict[c] = TargetEncoder(cols=c, smoothing=10, min_samples_leaf=5)

                #self.le_dict[c] = CatBoostEncoder(cols=c)

                normalLogger.debug('-- target encoder for %s' %c )
                tmp_df[c] = self.le_dict[c].fit_transform(tmp_df[c].astype(str), target)

                self.align_data = tmp_df[0:1] 
        else:
            for c in categorical_features:
                if c not in list(self.le_dict.keys()): #if new column, then skip it
                    continue

                #normalLogger.debug('-- target encoder for %s' %c )
                tmp_df[c] = self.le_dict[c].transform(tmp_df[c])

            _, tmp_df = self.align_data.align(tmp_df, join='left', axis=1, fill_value= -1)

        return tmp_df







    

