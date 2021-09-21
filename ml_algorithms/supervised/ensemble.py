import numpy as np

from collections import Counter
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForest:
    def __init__(self, n_trees, min_samples_split, max_depth):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self._predict_func = None

    def _bootstrap_sample(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)

        return x[idxs], y[idxs]

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [self._predict_func(tree_preds) for tree_preds in tree_preds]

        return np.array(y_pred)


class RandomForestClassifier(RandomForest):
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100):
        super().__init__(n_trees, min_samples_split, max_depth)
        self._predict_func = self._most_common_label

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth)
            x_sample, y_sample = self._bootstrap_sample(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]

        return most_common


class RandomForestRegressor(RandomForest):
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100):
        super().__init__(n_trees, min_samples_split, max_depth)
        self._predict_func = self._mean

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth)
            x_sample, y_sample = self._bootstrap_sample(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def _mean(self, neighbors_targets):
        mean = np.mean(neighbors_targets)
        
        return mean