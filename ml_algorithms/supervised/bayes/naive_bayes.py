import numpy as np


class NaiveBayesClassifier:
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for c in self._classes:
            x_c = x[c == y]
            self._mean[c, :] = x_c.mean(axis=0)
            self._var[c, :] = x_c.var(axis=0)
            self._priors[c] = x_c.shape[0] / float(n_samples)
            
    def predict(self, x):
        y_pred = [self._predict(sample) for sample in x]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)

        return numerator / denominator