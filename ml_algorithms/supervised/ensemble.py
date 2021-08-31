import numpy as np
from collections import Counter
from .tree import DecisionTree

def bootstrap_sample(x, y):
    n_samples = x.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return x[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                                max_depth=self.max_depth, n_feats=self.n_feats)
            x_sample, y_sample = bootstrap_sample(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [most_common_label(tree_preds) for tree_preds in tree_preds]
        return np.array(y_pred)