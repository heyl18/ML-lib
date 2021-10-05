from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from common.load_data import breast_cancer, load_data
X,y = load_data(breast_cancer, scale = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from linearSvm import LinearSVM
from knn import KNN
from logistic import Logistic


def evaluate(clf):
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    evaluate(KNN(5))
    evaluate(Logistic())
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    evaluate(LinearSVM())

    