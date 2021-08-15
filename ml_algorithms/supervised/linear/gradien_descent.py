import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x_b = np.c_[np.ones((n_samples, 1)), x]
        self.theta = np.random.randn(n_features+1, 1)
        y = y.reshape((n_samples, 1))

        for iteration in range(self.n_iterations):
            gradients = 2/n_samples * x_b.T.dot(x_b.dot(self.theta) - y)
            self.theta = self.theta - self.lr * gradients

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(self.theta)