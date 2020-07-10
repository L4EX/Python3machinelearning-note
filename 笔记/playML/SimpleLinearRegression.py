import numpy as np
from .metrics import r2_score
 
class SimpleLinearRegression1:
    def __init__(self):
         """初始化Simple Linear Regression 模型"""
         # 和knn有不同，knn算法需要保存训练集的特征和标签，当有测试用的数据传入的时候要和训练集中的特征进行计算，得到距离，实现预测
         # 而简单线性回归则不需要保存训练集的数据，值需要保存根据训练集得到的参数，在预测时使用的是参数，而不用训练集中的数据
         self.a_ = None
         self.b_ = None
    
    def fit(self, x_train, y_train):
        """根据训练集X_train, y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1,   \
            "Simple Linear Regression can only use solve single feature training data"
        assert len(x_train) == len(y_train),  \
            "the size of X_train must equal to the size of y_train"
    
        x_mean= np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x,y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        
        self.a_ = num / d
        self.b_ = y_mean -self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1,   \
            "Simple Linear Regression can only use solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegreesion1()"
    
class SimpleLinearRegression2:
    def __init__(self):
         """初始化Simple Linear Regression 模型"""
         # 和knn有不同，knn算法需要保存训练集的特征和标签，当有测试用的数据传入的时候要和训练集中的特征进行计算，得到距离，实现预测
         # 而简单线性回归则不需要保存训练集的数据，值需要保存根据训练集得到的参数，在预测时使用的是参数，而不用训练集中的数据
         self.a_ = None
         self.b_ = None
    
    def fit(self, x_train, y_train):
        """根据训练集X_train, y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1,   \
            "Simple Linear Regression can only use solve single feature training data"
        assert len(x_train) == len(y_train),  \
            "the size of X_train must equal to the size of y_train"
    
        x_mean= np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        # num = sum((x - x_mean) * (y - y_mean))
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = num / d
        self.b_ = y_mean -self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1,   \
            "Simple Linear Regression can only use solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """根据测试集 x_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(x_test)

        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegreesion2()"
    

