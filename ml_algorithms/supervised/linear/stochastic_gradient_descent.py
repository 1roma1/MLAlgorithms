import numpy as np

t0, t1 = 5, 50
def learning_shedule(t):
    return t0 / (t + t1)

class StochasticGradientDescent:
    def __init__(self, lr=0.1, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, x, y):
        self.theta = np.random.randn(x.shape[1], 1)
        y = y.reshape((x.shape[0], 1))
        m = x.shape[0]

        for epoch in range(self.n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = x[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.lr = learning_shedule(epoch * m + i)
                self.theta = self.theta - self.lr * gradients

    def predict(self, x):
        return x.dot(self.theta)