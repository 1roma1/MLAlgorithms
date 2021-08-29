import numpy as np


class LinearRegression:
    def __init__(self, alpha=0):
        self.weights = None
        self.alpha = alpha

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x_b = np.c_[np.ones((n_samples, 1)), x]
        reg_term = self.alpha*np.eye(n_features+1)
        reg_term[0, 0] = 0
        self.weights = np.linalg.inv(x_b.T.dot(x_b)+reg_term).dot(x_b.T).dot(y)

        return self
    
    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(self.weights)