import numpy as np
from sklearn.model_selection import train_test_split
from common.load_data import california_housing, load_data
from common.iterator import data_iterator
from common.plot import plot
X,y = load_data(california_housing, scale = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

'''
    使用d维平面 d为特征维数
    L(w,b,x,y) = ((xw + b) - y)^2 + lambda(w^2)
'''
class LinearRegression:
    def __init__(self, batch_size = 32, learning_rate = 0.001, max_epoch = 20, weight_reg = 0.2):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.weight_reg = weight_reg
 
    def __gradW(self,X,y):
        return 2 * X.T @ ((X @ self.w + self.b) - y) + 2 * self.weight_reg * self.w

    def __gradb(self,X,y):
        return 2 * np.sum((X @ self.w + self.b) - y)

    def __loss(self,X,y):
        return np.sum(((X @ self.w + self.b) - y)**2) + self.weight_reg * np.sum(self.w**2)

    def fit(self,X,y, need_plot = False):
        self.w = np.random.random(X.shape[1]).T
        self.b = np.random.random(1)
        loss = []
        acc = []
        for i in range(self.max_epoch):
            batch_loss = []
            batch_acc = []
            for X_batch, y_batch in data_iterator(X, y, self.batch_size, shuffle=True):
                batch_loss.append(self.__loss(X_batch, y_batch) / X_batch.shape[0])
                batch_acc.append(np.abs(self.predict(X_batch) - y_batch))
                gradW = self.__gradW(X_batch, y_batch)
                gradb = self.__gradb(X_batch, y_batch)
                self.w -= self.learning_rate * gradW / X_batch.shape[0] 
                self.b -= self.learning_rate * gradb / X_batch.shape[0] 
            print("Train epoch @", i, "mean_batch_diff: %.4f" % np.mean(batch_acc), "mean_batch_loss: %.4f" % np.mean(batch_loss))
            if need_plot:
                loss.append(np.mean(batch_loss))
                acc.append(np.mean(batch_acc))
        if need_plot:
            plot(range(self.max_epoch), acc, label="Diff")
            plot(range(self.max_epoch), loss, label="Loss")

    def predict(self,X):
        return (X @ self.w) + self.b

reg = LinearRegression()
reg.fit(X_train, y_train, need_plot = True)
print(np.sum(np.abs(reg.predict(X_test)-y_test))/y_test.shape[0])

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print(np.sum(np.abs(reg.predict(X_test)-y_test))/y_test.shape[0])