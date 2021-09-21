import numpy as np


class LinearRegression:
    def __init__(self, l2=0.01):
        self.weights = None
        self.l2 = l2

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x = np.c_[np.ones((n_samples, 1)), x]

        reg_term = self.l2 * np.eye(n_features + 1)
        reg_term[0, 0] = 0

        self.weights = np.linalg.inv(x.T.dot(x) + reg_term).dot(x.T).dot(y)

        return self

    def predict(self, x):
        x = np.c_[np.ones((x.shape[0], 1)), x]

        return x.dot(self.weights)


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x = np.c_[np.ones((n_samples, 1)), x]

        self.weights = np.zeros(n_features+1)

        for _ in range(self.n_iters):
            y_pred = self._sigmoid(np.dot(x, self.weights))
            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            self.weights -= self.lr * dw

    def predict(self, x):
        x = np.c_[np.ones((x.shape[0], 1)), x]

        y_pred = self._sigmoid(np.dot(x, self.weights))
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        
        return y_pred_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class GradientDescent:
    def __init__(self, lr=0.01, n_iterations=1000, penalty=None, alpha=0):
        self.weights = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, x, y):
        n_samples, n_features = x.shape
        x = np.c_[np.ones((n_samples, 1)), x]

        self.weights = np.random.randn(n_features+1, 1)
        y = y.reshape((n_samples, 1))

        grad_func = self._get_grad_func()

        for _ in range(self.n_iterations):
            grad = grad_func(x, y)
            self.weights = self.weights - self.lr * grad

    def _get_grad_func(self):
        def l1_grads(x, y):
            return 2/x.shape[0] * (x.T.dot(x.dot(self.weights) - y)+self.alpha*np.sign(self.weights))

        def l2_grads(x, y):
            return 2/x.shape[0] * (x.T.dot(x.dot(self.weights) - y)+2*self.alpha*self.weights)

        def without_penalty_grads(x, y):
            return 2/x.shape[0] * x.T.dot(x.dot(self.weights) - y)

        if self.penalty == 'l1':
            return l1_grads
        elif self.penalty == 'l2':
            return l2_grads
        else:
            return without_penalty_grads

    def predict(self, x):
        x = np.c_[np.ones((x.shape[0], 1)), x]

        return x.dot(self.weights)


class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=50, l2=0.0, minibatches=1):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches      

    def fit(self, x, y):
        self.n_classes = np.max(y) + 1
        self.n_features = x.shape[1]

        x = np.c_[np.ones((x.shape[0], 1)), x]
        y = self._one_hot(y, self.n_classes)

        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(self.n_features+1, self.n_classes))

        for _ in range(self.epochs):
            for idx in self._get_minibatches_idx(self.minibatches, y):
                net = x[idx].dot(self.weights)
                softm = self._softmax(net)
                diff = softm - y[idx]

                grad = np.dot(x[idx].T, diff)

                self.weights -= (self.lr * grad + self.lr * self.l2 * self.weights)

        return self
    
    def _predict(self, x):
        probas = self.predict_proba(x)
        return probas.argmax(axis=1)

    def predict(self, X):
        return self._predict(X)

    def predict_proba(self, x):
        x = np.c_[np.ones((x.shape[0], 1)), x]

        net = x.dot(self.weights) 
        softm = self._softmax(net)

        return softm

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
    
    def _one_hot(self, y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(np.float)    
    
    def _get_minibatches_idx(self, n_batches, y):
            indices = np.arange(y.shape[0])
            indices = np.random.permutation(indices)

            if n_batches > 1:
                remainder = y.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch