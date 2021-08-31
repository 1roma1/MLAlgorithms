import numpy as np


class LinearRegression:
    def __init__(self, alpha=0):
        self.weights = None
        self.alpha = alpha

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x_b = np.c_[np.ones((n_samples, 1)), x]
        reg_term = self.alpha * np.eye(n_features + 1)
        reg_term[0, 0] = 0
        self.weights = np.linalg.inv(x_b.T.dot(x_b) + reg_term).dot(x_b.T).dot(y)

        return self

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b.dot(self.weights)


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_predicted = self._sigmoid(np.dot(x, self.weights) + self.bias)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


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