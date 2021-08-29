import numpy as np

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


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, x, y):
        n_samples, n_features = x.shape

        w = np.full(n_samples, (1/n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                x_column = x[:, feature_i]
                thresholds = np.unique(x_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[x_column < threshold] = -1

                    missclassified = w[y != predictions]
                    error = sum(missclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

        EPS = 1e-10
        clf.alpha = 0.5*np.log((1-error) / (error+EPS))

        predictions = clf.predict(x)

        w *= np.exp(-clf.alpha * y * predictions)
        w /= np.sum(w)

        self.clfs.append(clf)

    def predict(self, x):
        clf_preds = [clf.alpha*clf.predict(x) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred
