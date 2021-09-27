import logging
import argparse
import pandas as pd

def get_logger(loggerName, level, filename):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    level = levels[level]

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s  %(levelname).8s  %(message)s')
    file_handler = logging.FileHandler(filename, mode = 'w') #create new file
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = get_logger('logger', 'debug', 'CheckerLogger.log')


class check_quality(object):
    def __init__(self, df):
        self.df = df
        
    def null_checker(self):
        null_summary = pd.DataFrame({
            'count': len(self.df),
            'null_count': self.df.isnull().sum(),
            'null_ratio(%)': self.df.isnull().sum()/len(df)*100,
        })
        #self.summary['null_checker'] = null_summary
        
        logger.debug('='*20 + 'null checker' + '='*20)
        logger.debug(null_summary)
        logger.debug('\n')
        
    
    def id_checker(self, id_col): #id column should be unique and complete
        logger.debug('='*20 + 'id checker' + '='*20)
        # complete
        if self.df[id_col].isnull().sum()>0:
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
            
        
        
    def label_checker(self, label_col, label_list):
        logger.debug('='*20 + 'label checker' + '='*20)
        # complete
        if self.df[label_col].isnull().sum()>0:
            #self.summary['label_checker'] = '[Warning!] label column:{} should not contain Nulls'.format(label_col)
            logger.debug('[Warning!] label column:{} should not contain Nulls'.format(label_col))
        else:
            #self.summary['label_checker'] = 'label column:{} has no Nulls'.format(label_col)
            logger.debug('label column:{} has no Nulls'.format(label_col))
        
        # element check
        wrong_label = []
        for i in set(self.df[label_col]):
            if i not in label_list:
                wrong_label.append(i)
        if len(wrong_label)>0:
            #self.summary['label_checker'] += '\n[Warning!] label column:{} has elements not in label list:{}'.format(label_col, wrong_label)
            logger.debug('[Warning!] label column:{} has elements not in label list:{}'.format(label_col, wrong_label))
        else:
            #self.summary['label_checker'] += '\nlabel column element check ok!'
            logger.debug('label column element check ok!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='csv file basic data quality checker')
    parser.add_argument('--data_dir', help='path to data')
    parser.add_argument('--id_col', help='path to data')
    parser.add_argument('--label_col', help='path to data')
    parser.add_argument('--label_list', type=str, help='path to data')
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir)
    checker = check_quality(df)

    checker.null_checker()
    checker.id_checker(args.id_col)
    checker.label_checker(args.label_col, [int(i) for i in args.label_list.split(',')])





  