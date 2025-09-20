import numpy as np

class LinearRegression_OrdinaryLeastSquares:
    def __init__(self):
        self.w = None  # веса

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # добавляем столбец единиц для свободного члена
        X = np.c_[np.ones(X.shape[0]), X]
        # формула OLS
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w


# --- пример ---
X = [[1,2],
     [2,1],
     [3,5],
     [4,7]]
y = [10, 8, 20, 25]

model = LinearRegression_OrdinaryLeastSquares()
model.fit(X, y)
print("Веса:", model.w)
print("Прогноз:", model.predict([[5, 6], [6, 7]]))