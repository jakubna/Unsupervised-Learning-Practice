from algorithms.kmeans import Kmeans
import numpy as np


class Kpifs(Kmeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _update_clusters(self, x: np.ndarray):
        """
        Update the cluster assignment.
        :param x: 2D data array of size (rows, features).
        :return delta: Ckusters indicators.
        :return nearest_id: List of nearest observations' indices for each cluster.
        :return clusters: Cluster indexes assigned to each observation (labels for each point).
        """
        distances = np.zeros(shape=(x.shape[0], self.M))
        nearest_id = [[] for _ in range(self.M)]
        delta = np.zeros(shape=(x.shape[0], self.M))
        clusters = []
        for row_id, row in enumerate(x):
            for centroid_id, centroid in enumerate(self.centroids):
                d = self._calculate_distance(centroid, row)
                Ck_size = len(nearest_id[centroid_id])
                if Ck_size > 0:
                    distances[row_id, centroid_id] = d * Ck_size
                else:
                    distances[row_id, centroid_id] = d
            cluster_id = int(np.argmin(distances[row_id, :]))
            clusters.append(cluster_id)
            nearest_id[cluster_id].append(row_id)
            delta[row_id, cluster_id] = 1
        return delta, nearest_id, clusters
