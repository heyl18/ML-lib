import numpy as np
from common.iterator import data_iterator
from common.plot import plot

'''
    以下为二分类问题的SVM, label = -1 || 1
'''
class LinearSVM:
    def __init__(self, learning_rate = 0.01, w_reg = 0.1, max_epoch = 30, batch_size = 32):
        self.learning_rate = learning_rate
        self.w_reg = w_reg
        self.max_epoch = max_epoch
        self.batch_size = batch_size

    def __loss(self,X,y):
        hinge_loss = np.maximum(0, 1 - (y * ( (X @ self.w ) + self.b)))
        loss = np.mean(hinge_loss) + 0.5 * self.w_reg * np.sum(self.w**2)
        dW = self.w_reg * self.w - np.where(hinge_loss > 0, y, 0) @ X
        db = - np.mean(y * (hinge_loss > 0),axis=0)
        return loss, dW , db

    def fit(self, X, y, update_epoch = 5, need_plot=False, verbose = False):
        '''
            trick: 每次训练update_epoch后，使用平均值来更新w和b
        '''
        self.w = np.random.randn(X.shape[1])
        self.b = np.random.randn(1)
        self.meanW = np.zeros_like(self.w)
        self.meanb = np.zeros_like(self.b)
        acc = []
        loss = []
        for i in range(self.max_epoch):
            batch_loss = []
            batch_acc = []
            for X_batch, y_batch in data_iterator(X, y, self.batch_size, shuffle=True):
                current_loss, gradW, gradb = self.__loss(X_batch, y_batch)
                batch_loss.append(current_loss)
                batch_acc.append(np.sum(self.predict(X_batch) == y_batch) / X_batch.shape[0])
                self.w -= self.learning_rate * gradW
                self.b -= self.learning_rate * gradb
            if verbose:
                print("Train epoch @", i, "mean_batch_acc: %.4f" % np.mean(batch_acc), "mean_batch_loss: %.4f" % np.mean(batch_loss))
            self.meanW += self.w / update_epoch
            self.meanb += self.b / update_epoch
            if i % update_epoch == 0:
                self.w = self.meanW
                self.b = self.meanb
                self.meanW = np.zeros_like(self.w)
                self.meanb = np.zeros_like(self.b)
            if need_plot:
                loss.append(np.mean(batch_loss))
                acc.append(np.mean(batch_acc))
        if need_plot:
            plot(range(self.max_epoch), acc)
            plot(range(self.max_epoch), loss, label="Loss")
    
    def predict(self, X):
        y_pred = (X @ self.w) + self.b
        y_pred[y_pred <= 0] = -1
        y_pred[y_pred > 0] = 1
        return y_pred

if __name__ == '__main__':
    from common.load_data import breast_cancer, load_data
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    X,y = load_data(breast_cancer, scale = True)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    clf = LinearSVM()
    clf.fit(X_train, y_train, need_plot = True)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    from sklearn.svm import LinearSVC as s
    clf = s()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))