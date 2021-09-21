import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        x = x - self.mean

        cov = np.cov(x.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[0:self.components]

    def transform(self, x):
        x = x - self.mean
        return np.dot(x, self.components.T)