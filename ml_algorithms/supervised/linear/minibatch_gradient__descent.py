import numpy as np

t0, t1 = 200, 1000
def learning_shedule(t):
    return t0 / (t + t1)

class MinibatchGradientDescent:
    def __init__(self, batch_size, lr=0.1, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, x, y):
        self.theta = np.random.randn(x.shape[1], 1)
        y = y.reshape((x.shape[0], 1))
        m = x.shape[0]
        t = 0
        for epoch in range(self.n_epochs):
            shuffled_indices = np.random.permutation(m)
            x_shuffled = x[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for i in range(0, m, self.batch_size):
                t += 1
                xi = x_shuffled[i:i+self.batch_size]
                yi = y_shuffled[i:i+self.batch_size]
                gradients = 2/self.batch_size * xi.T.dot(xi.dot(self.theta) - yi)
                self.lr = learning_shedule(t)
                self.theta = self.theta - self.lr * gradients

    def predict(self, x):
        return x.dot(self.theta)