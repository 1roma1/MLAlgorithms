import numpy as np

from .tree import DecisionTreeRegressor

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, x):
        n_samples = x.shape[0]
        x_column = x[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[x_column < self.threshold] = -1
        else:
            predictions[x_column > self.threshold] = -1
            
        return predictions


class AdaBoostClassifier:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.5, min_samples_split=2, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTreeRegressor(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self._gradient_func(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)

            y_pred -= self.learning_rate * update

    def _gradient_func(self, y, y_pred):
        return -(y - y_pred)

    def predict(self, X):
        y_pred = np.array([])

        for tree in self.trees:
            update = tree.predict(X)
            update = self.learning_rate * update
            y_pred = -update if not y_pred.any() else y_pred - update

        return y_pred