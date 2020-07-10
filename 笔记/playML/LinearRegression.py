import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression
        \theta中的截距和系数要区别保存"""

        self.coef_ = None # 参数（向量）
        self.interception_ = None # 截距（数值）
        self._theta = None # 整体\theta
    
    def fit_normal(self, X_train, y_train):
        #正规方程解
        """根据训练集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta =  np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        #sklearn中，fit()函数返回self对象本身
        return self

    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4):
        """根据训练数据集X_train， y_train，使用梯度下降法训练Linear Regression模型"""
        # 没有spsilon的参数默认值
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum( X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return 2 * res / len(X_b)


            #向量化运算求梯度
            return 2 * X_b.T.dot(X_b.dot(theta) - y) / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon= 1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient_descent = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient_descent
                if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train, n_iters = 1e4, t0 =5, t1 = 50):
        #循环次数应该不少于样本总数，这样才能把所有的样本信息都利用到，这样一来，n_iters的含义可以变成循环几遍全体样本
        assert n_iters >= 1, \
            "n_iters must bigger than 1"
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2 
        def sgd(X_b, y, initial_theta, n_iters, t0 = 5, t1 = 50):
            #不需要手动设置eta
            #t0也就是a
            #t1也就是b
    
            def learning_rate(t):
                return t0/(t + t1)
    
            theta = initial_theta
            #gd中，搜索退出的条件有两个，一个是迭代次数达到上限，一个是前后两次损失差小于一个很小的值。
            #但是在sgd中，因为每次搜索都是随机的，所以前后两次损失差小于一个很小的值也不一定就是找到了极小值
            #所以只有一个搜索次数的退出条件
            #因为不需要计算前后两次的损失差，所以J()函数也不用了
            m = len(X_b)

            for cur_iter in range(n_iters):
                #既然希望把所有样本都能遍历到，如果只是单纯的选随机数，可能有的样本还是遍历不到
                #不难想到，可以生成一个乱序数组，每次迭代一遍，就按这个数组的排序选样本，就能遍历到全部样本了
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]

                #随机选择一个样本
                # rand_i = np.random.randint(m)

                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate( cur_iter * m +  i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta,n_iters, t0, t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    def predict(self, X_predict):
        """给定待预测数据集X_predict, 返回表示X_predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        x_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return  x_b.dot(self._theta)
    
    def score(self, X_test, y_test):
        """根据测试集X_test和y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)


    def __repr__(self):
        return "LinearRegression()"