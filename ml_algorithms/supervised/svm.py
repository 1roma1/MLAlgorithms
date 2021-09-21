import numpy as np


class LinearSVM:
    def __init__(self, C=1, lr=0.01, n_epochs=1000):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, x, y):
        w = np.random.randn(x.shape[1], 1)
        b = 0

        t = (y * 2 - 1).reshape(y.shape[0], 1)
        x_t = x * t

        for _ in range(self.n_epochs):
            support_vectors_idx = (x_t.dot(w) + t * b < 1).ravel()
            x_t_sv = x_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            w_grad = w - self.C * np.sum(x_t_sv, axis=0).reshape(-1, 1)
            b_grad = -self.C * np.sum(t_sv)

            w = w - self.lr * w_grad
            b = b - self.lr * b_grad

        self.bias = np.array([b])
        self.weights = np.array([w])
        support_vectors_idx = (x_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = x[support_vectors_idx]

        return self

    def _decision_function(self, x):
        return x.dot(self.weights[0]) + self.bias[0]

    def predict(self, x):
        return (self._decision_function(x) >= 0).astype(np.float64)