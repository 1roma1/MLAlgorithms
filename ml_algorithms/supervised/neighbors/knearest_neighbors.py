import numpy as np

class KNearestNeighbor:
    def __init__(self, neighbors=1):
        self.neighbors = neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def predict(self, x):
        num_samples = x.shape[0]
        y_pred = np.zeros(num_samples, dtype=self.y.dtype)

        for i in range(num_samples):
            distances = np.sum(np.abs(self.x - x[i, :]), axis=1)
            indices = np.argsort(distances)[:self.neighbors]
            y_pred[i] = np.sum(self.y[indices]) / self.neighbors

        return y_pred