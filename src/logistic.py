import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from common.load_data import breast_cancer, load_data
from common.iterator import data_iterator
from common.plot import plot
X,y = load_data(breast_cancer, scale = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

'''
    注意：以下为二分类问题的逻辑回归
'''
class Logistic:
    def __init__(self, batch_size = 32, learning_rate = 0.1, max_epoch = 30, weight_reg = 0.2):
        self.learning_rate = learning_rate
        self.weight_reg = weight_reg
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.sigmoid = lambda x: 1.0/(1.0+np.exp(-x))

    def fit(self, X, y, need_plot = False):
        self.w = np.zeros(X.shape[1])
        loss = []
        acc = []
        for i in range(self.max_epoch):
            batch_loss = []
            batch_acc = []
            for X_batch, y_batch in data_iterator(X, y, self.batch_size, shuffle=True):
                batch_loss.append(self.__loss(X_batch, y_batch) / X_batch.shape[0])
                batch_acc.append(np.sum(self.predict(X_batch) == y_batch) / X_batch.shape[0])
                grad = self.__grad(X_batch, y_batch)
                self.w -= self.learning_rate * grad / X_batch.shape[0]
            print("Train epoch @", i, "mean_batch_acc: %.4f" % np.mean(batch_acc), "mean_batch_loss: %.4f" % np.mean(batch_loss))
            if need_plot:
                loss.append(np.mean(batch_loss))
                acc.append(np.mean(batch_acc))
        if need_plot:
            plot(range(self.max_epoch), acc)
            plot(range(self.max_epoch), loss, label="Loss")
            
    def __loss(self, X, y):
        h = self.sigmoid(X @ self.w)
        return -np.sum(y @ np.log(h) + (1-y) @ np.log(1-h)) + self.weight_reg * np.sum(self.w**2)

    def __grad(self, X, y):
        return X.T @ (self.sigmoid(X @ self.w) - y)

    def predict(self, X):
        predict = self.sigmoid(X @ self.w)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        return predict

clf = Logistic()
clf.fit(X_train,y_train, need_plot=True)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))