import numpy as np

from scipy.spatial.distance import euclidean


class KMeans:
    def __init__(self, n_clusters=5, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.n_clusters)]
        self.centroids = []
    
    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.centroids = [self.x[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels
            
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)

        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.n_clusters, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean(centroids_old[i], centroids[i]) for i in range(self.n_clusters)]
        
        return sum(distances) == 0