import logging
import argparse
import pandas as pd
import util

pd.set_option('display.max_rows', 500)


logger = util.get_logger('logger', 'debug', './logs/CheckerLogger.%Y-%m-%d.log')

class check_quality(object):
    def __init__(self, df):
        self.df = df
        
    def null_checker(self):
        null_summary = pd.DataFrame({
            'count': len(self.df),
            'null_count': self.df.isnull().sum(),
            'null_ratio(%)': self.df.isnull().sum() / len(self.df) * 100,
        })
        #self.summary['null_checker'] = null_summary
        
        logger.debug('=' * 20 + 'null checker' + '=' * 20)
        logger.debug('null summary:\n{}\n'.format(null_summary.to_string()))
        
    
    def id_checker(self, id_col): #id column should be unique and complete
        logger.debug('=' * 20 + 'id checker' + '=' * 20)
        # complete
        if self.df[id_col].isnull().sum() > 0:
            #self.summary['id_checker'] = '[Warning!] id column:{} should not contain Nulls\n'.format(id_col)
            logger.debug('[Warning!] id column:{} should not contain Nulls'.format(id_col))
        else:
            #self.summary['id_checker'] = 'id column:{} has no Nulls\n'.format(id_col)
            logger.debug('id column:{} has no Nulls'.format(id_col))
        
        # unique
        if len(set(self.df[id_col])) != len(self.df):
            tmp = self.df.groupby(id_col)[id_col].count()
            #self.summary['id_checker'] += '[Warning!] id column:{} should should be unique. {} is dulplicated!'.format(id_col, tmp[tmp>1].index.tolist())
            logger.debug('[Warning!] id column:{} should should be unique. {} is dulplicated!\n'.format(id_col, tmp[tmp>1].index.tolist()))
        else:
            #self.summary['id_checker'] += 'id column:{} is unique'.format(id_col)
            logger.debug('id column:{} is unique\n'.format(id_col))
            
    def categorical_checker(self):
        logger.debug('=' * 20 + 'categorical checker checker' + '=' * 20)
        col_name, class_cnt = [],[]
        for c in self.df.columns:
            if str(self.df[c].dtypes)=='object':
                col_name.append(c)
                class_cnt.append( len(set(self.df[c]) ))
        cnt_summary = pd.DataFrame({'column':col_name, 'class_cnt': class_cnt})
        logger.debug('categorical column class count:\n{}\n'.format(cnt_summary.to_string()))
        #logger.debug(pd.DataFrame(class_cnt_dict,index=[0]))

        
    def label_checker(self, label_col, label_list=None):
        logger.debug('=' * 20 + 'label checker' + '=' * 20)
        # complete
        if self.df[label_col].isnull().sum()>0:
            logger.debug('[Warning!] label column:{} should not contain Nulls\n'.format(label_col))
        else:
            logger.debug('label column:{} has no Nulls\n'.format(label_col))
        
        # element check
        if label_list is not None:
            if isinstance(label_list, str):
                label_list = [int(i) for i in label_list.split(',')]
            
            wrong_label = []
            for i in set(self.df[label_col]):
                if i not in label_list:
                    wrong_label.append(i)
            if len(wrong_label) > 0:
                logger.debug('[Warning!] label column:{} has elements not in label list:{}\n'.format(label_col, wrong_label))
            else:
                logger.debug('label column element check ok!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='csv file basic data quality checker')
    parser.add_argument('--data_dir', help='path to data')
    parser.add_argument('--id_col', help='path to data')
    parser.add_argument('--label_col', help='path to data')
    parser.add_argument('--label_list', default=None, type=str, help='path to data')
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir)
    checker = check_quality(df)

    checker.null_checker()
    checker.categorical_checker()

    if args.id_col is not None:
        checker.id_checker(args.id_col)

    if args.label_col is not None:
        checker.label_checker(args.label_col, args.label_list)





  