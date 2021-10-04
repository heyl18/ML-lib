from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from common.load_data import iris, load_data
X, y = load_data(iris, scale = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

class KNN:
    def __init__(self, K):
        self.K = K
        self.distance = lambda x,y: \
            np.sqrt(np.sum((x-y)**2))
    
    def fit(self, X, y):
        self.trainX = X
        self.trainy = y

    def __predict(self, x):
        distances = [self.distance(x,trainx) for trainx in self.trainX]
        k_indices = np.argsort(distances)[:self.K]
        k_nearest_labels = [self.trainy[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        predicted_labels = [self.__predict(x) for x in X]
        return np.array(predicted_labels)

if __name__ == '__main__':
    model = KNN(5)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))