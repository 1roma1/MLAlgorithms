import numpy as np

class GradientDescent:
    def __init__(self, lr=0.1, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations

    def fit(self, x, y):
        self.theta = np.random.randn(x.shape[1], 1)
        y = y.reshape((x.shape[0], 1))
        m = x.shape[0]

        for iteration in range(self.n_iterations):
            gradients = 2/m * x.T.dot(x.dot(self.theta) - y)
            self.theta = self.theta - self.lr * gradients

    def predict(self, x):
        return x.dot(self.theta)