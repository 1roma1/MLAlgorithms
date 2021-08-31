import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter

class KNNClassifier:
    def __init__(self, neighbors=1):
        self.neighbors = neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def predict(self, x):
        y_pred = [self._predict(sample) for sample in x]
        return y_pred

    def _predict(self, x):
        distances = (euclidean(x, sample) for sample in self.x)
        neighbors = sorted(((dist, target) for (dist, target) in zip(distances, self.y)), key=lambda x: x[0])
        neighbors_targets = [target for (_, target) in neighbors[:self.neighbors]]

        most_common_label = Counter(neighbors_targets).most_common(1)[0][0]

        return most_common_label


class KNNRegressor:
    def __init__(self, neighbors=1):
        self.neighbors = neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        y_pred = [self._predict(sample) for sample in x]
        return y_pred

    def _predict(self, x):
        distances = (euclidean(x, sample) for sample in self.x)
        neighbors = sorted(((dist, target) for (dist, target) in zip(distances, self.y)), key=lambda x: x[0])
        neighbors_targets = [target for (_, target) in neighbors[:self.neighbors]]

        mean = np.mean(neighbors_targets)

        return mean