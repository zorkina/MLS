import numpy as np

class LinearRegression_OrdinaryLeastSquares:
    def __init__(self):
        self.w = None # weight


    def fit(self, X, y, w):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.invert(X @ X.T) @ X.T @ y


    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w

X = [[1,2], [2,3], [3,4], [4,5]]
LinearRegression_OrdinaryLeastSquares.predict(X)

