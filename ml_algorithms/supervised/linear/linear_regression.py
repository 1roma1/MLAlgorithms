import numpy as np

class LinearRegression:
    def fit(self, x, y):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        self.weights = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
        return self
    
    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(self.weights)