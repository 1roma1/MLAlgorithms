import numpy as np


class GradientDescent:
    def __init__(self, lr=0.01, n_iterations=1000, penalty=None, alpha=0):
        self.theta = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x_b = np.c_[np.ones((n_samples, 1)), x]
        self.theta = np.random.randn(n_features+1, 1)
        y = y.reshape((n_samples, 1))

        grad_func = self._get_gradients()

        for iteration in range(self.n_iterations):
            gradients = grad_func(x_b, y)
            self.theta = self.theta - self.lr * gradients

    def _get_gradients(self):
        def l1_grads(x, y):
            return 2/x.shape[0] * x.T.dot(x.dot(self.theta) - y)+self.alpha*np.sign(self.theta)

        def l2_grads(x, y):
            return 2/x.shape[0] * x.T.dot(x.dot(self.theta) - y)+2*self.alpha*self.theta

        def without_penalty_grads(x, y):
            return 2/x.shape[0] * x.T.dot(x.dot(self.theta) - y)

        if self.penalty == 'l1':
            return l1_grads
        elif self.penalty == 'l2':
            return l2_grads
        else:
            return without_penalty_grads

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(self.theta)