import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict)/len(y_predict)

def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must equal to the size of y_predict"
    return np.sum((y_predict - y_true) ** 2) / len(y_predict)

def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict的RMSE"""

    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict的MAE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must equal to the size of y_predict"

    return np.sum(np.absolute(y_predict - y_true)) / len(y_predict)

def r2_score(y_true, y_predict):
    return 1 - mean_absolute_error(y_true, y_predict) / np.var(y_true)