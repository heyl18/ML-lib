import numpy as np
from common.iterator import data_iterator
from common.plot import plot

'''
    使用d维平面 d为特征维数
    L(w,b,x,y) = 0.5 * ((xw + b) - y)^2 + 0.5 * lambda(w^2)
'''
class LinearRegression:
    def __init__(self, batch_size = 64, learning_rate = 0.0001, max_epoch = 100, weight_reg = 1.5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.weight_reg = weight_reg

    def __loss(self,X,y):
        pred = (X @ self.w + self.b) - y
        loss = 0.5 * np.sum(pred**2) / X.shape[0] + 0.5 * self.weight_reg * np.sum(self.w**2)
        dw = X.T @ (pred - y) / X.shape[0] + self.weight_reg * self.w
        db = np.sum(pred) / X.shape[0]
        return loss, dw, db

    def fit(self,X,y, need_plot = False, verbose = False):
        self.w = np.random.random(X.shape[1]).T
        self.b = np.random.random(1)
        loss = []
        acc = []
        for i in range(self.max_epoch):
            batch_loss = []
            batch_acc = []
            for X_batch, y_batch in data_iterator(X, y, self.batch_size, shuffle=True):
                current_loss, dw, db = self.__loss(X_batch, y_batch)
                batch_loss.append(current_loss)
                batch_acc.append(np.mean(np.abs(self.predict(X_batch) - y_batch)))
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            if verbose:
                print("Train epoch @", i, "mean_batch_diff: %.4f" % np.mean(batch_acc), "mean_batch_loss: %.4f" % np.mean(batch_loss))
            if need_plot:
                loss.append(np.mean(batch_loss))
                acc.append(np.mean(batch_acc))
        if need_plot:
            plot(range(self.max_epoch), acc, label="Diff")
            plot(range(self.max_epoch), loss, label="Loss")

    def predict(self,X):
        return (X @ self.w) + self.b

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from common.load_data import california_housing, load_data
    X,y = load_data(california_housing, scale = True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    reg = LinearRegression()
    reg.fit(X_train, y_train, need_plot = True)
    print(np.mean(np.abs(reg.predict(X_test)-y_test)))

    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    print(np.mean(np.abs(reg.predict(X_test)-y_test)))