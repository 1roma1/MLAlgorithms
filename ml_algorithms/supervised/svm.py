import numpy as np


class LinearSVM:
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, x, y):
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(x.shape[1], 1)
        b = 0

        m = len(x)
        t = y * 2 - 1
        x_t = x * t
        self.Js = []

        for epoch in range(self.n_epochs):
            support_vectors_idx = (x_t.dot(w) + t * b < 1).ravel()
            x_t_sv = x_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1/2 * np.sum(w*w) + self.C * (np.sum(1 - x_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)

            w_gradient_vector = w - self.C * np.sum(x_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -self.C * np.sum(t_sv)

            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (x_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = x[support_vectors_idx]
        return self

    def decision_function(self, x):
        return x.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, x):
        return (self.decision_function(x) >= 0).astype(np.float64)