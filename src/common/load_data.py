from sklearn import datasets, preprocessing

# 2-class clf
breast_cancer = datasets.load_breast_cancer()

# multi-class clf
iris = datasets.load_iris()

# regression
california_housing = datasets.fetch_california_housing()

diabetes = datasets.load_diabetes()


def load_data(dataset, scale = False, normalize = False):
    '''
        For sklearn datasets, we can use this function to get the X and y
    '''
    X,y = dataset.data, dataset.target
    if scale:
        X = preprocessing.scale(X)
    if normalize:
        X = preprocessing.normalize(X)
    return X,y