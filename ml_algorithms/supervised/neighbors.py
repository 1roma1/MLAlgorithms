import numpy as np

from collections import Counter
from scipy.spatial.distance import euclidean


class KNN:
    def __init__(self, neighbors):
        self.neighbors = neighbors
        self._predict_func = None

    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def predict(self, x):
        y_pred = [self._predict(sample) for sample in x]
        return y_pred

    def _predict(self, x):
        distances = (euclidean(x, sample) for sample in self.x)
        neighbors = sorted(((dist, target) for (dist, target) in zip(distances, self.y)))
        neighbors_targets = [target for (_, target) in neighbors[:self.neighbors]]

        prediction = self._predict_func(neighbors_targets)
        

        return prediction


class KNNClassifier(KNN):
    def __init__(self, neighbors=1):
        super().__init__(neighbors)
        self._predict_func = self._most_common_label

    def _most_common_label(self, neighbors_targets):
        most_common_label = Counter(neighbors_targets).most_common(1)[0][0]

        return most_common_label


class KNNRegressor(KNN):
    def __init__(self, neighbors=1):
        super().__init__(neighbors)
        self._predict_func = self._mean

    def _mean(self, neighbors_targets):
        mean = np.mean(neighbors_targets)

        return mean