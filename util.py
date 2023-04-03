import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
import threading
from logging.handlers import BaseRotatingHandler
from matplotlib import pyplot as plt
import seaborn as sns
import time
import codecs

from sklearn.metrics import (confusion_matrix, roc_auc_score, accuracy_score,
                            recall_score, precision_score, mean_squared_error, mean_absolute_percentage_error)


plt.style.use('seaborn')


def folder_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def folder_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_confusion_table(y_true, y_pred, cat, save_path=None):
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)

    # graph confusion table and save
    sns.set(font_scale=1.3)
    plt.figure()
    fig_plot = sns.heatmap(conf, cmap=colormap, annot=True, cbar=False, fmt='d')
    fig = fig_plot.get_figure()
    plt.title('%s auc: %.3f, acc:%.3f, \nrecall:%s, precision:%s' % (cat, auc, acc, str(round(recall, 3)),
                                                                     str(round(precision, 3))))
    if save_path:
        fig.savefig(os.path.join(save_path, "%s_confusion.png" % cat))
        plt.close()
    else:
        plt.show()


def get_scatter_eval(y_true, y_pred, cat, save_path=None):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    corr = np.corrcoef(list(y_true), list(y_pred))[0][1]
    
    # graph scatterplot table and save
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.title("MAPE:{:.3f}, RMSE:{:.3f}, correlation:{:.3f}".format(mape, rmse, corr))
    plt.xlabel("true value")
    plt.ylabel("predict value")
    if save_path:
        plt.savefig(os.path.join(save_path, "%s_scatter.png" % cat))
        plt.close()
    else:
        plt.show()


def get_prob_plot(y_true, pred_prob, cat, save_path=None):
    result_df = pd.DataFrame({'prob': pred_prob, 'cat': y_true})

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=result_df, x='prob', hue='cat', kde=True)
    plt.legend(['larger than threshold', 'less than threshold'])
    plt.subplot(1, 2, 2)
    sns.boxplot(data=result_df, x='cat', y='prob').set(
        xlabel='label'
    )
    if save_path:
        plt.savefig(os.path.join(save_path, "%s_prob.png" % cat))
        plt.close()
    else:
        plt.show()


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

    logfilename = datetime.now().strftime(filename)
    file_handler = logging.FileHandler(logfilename, 'a', 'utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # time_rotate_handler = TimedRotatingFileHandler(logfilename, when='midnight', backupCount=365)
    # time_rotate_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # logger.addHandler(time_rotate_handler)

    return logger


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def MultiProcessLogger(loggerName, filename):
    logger = logging.getLogger(loggerName)

    level = logging.INFO
    format = '%(asctime)s %(levelname)-8s %(message)s'
    logfilename = datetime.now().strftime(filename)
    hdlr = MultiProcessSafeDailyRotatingFileHandler(logfilename, encoding='utf-8')
    fmt = logging.Formatter(format)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    logger.setLevel(level)

    return logger


# ref: https://my.oschina.net/lionets/blog/796438
class MultiProcessSafeDailyRotatingFileHandler(BaseRotatingHandler):
    """Similar with `logging.TimedRotatingFileHandler`, while this one is
    - Multi process safe
    - Rotate at midnight only
    - Utc not supported
    """
    def __init__(self, filename, encoding=None, delay=False, utc=False, **kwargs):
        self.utc = utc
        self.suffix = "%Y-%m-%d.log"
        self.baseFilename = filename
        self.currentFileName = self._compute_fn()
        BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)

    def shouldRollover(self, record):
        if self.currentFileName != self._compute_fn():
            return True
        return False

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.currentFileName = self._compute_fn()

    def _compute_fn(self):
        return self.baseFilename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        if self.encoding is None:
            stream = open(self.currentFileName, self.mode)
        else:
            stream = codecs.open(self.currentFileName, self.mode, self.encoding)
        # simulate file name structure of `logging.TimedRotatingFileHandler`
        if os.path.exists(self.baseFilename):
            try:
                os.remove(self.baseFilename)
            except OSError:
                pass
        try:
            os.symlink(self.currentFileName, self.baseFilename)
        except OSError:
            pass
        return stream


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)

        return self._return
