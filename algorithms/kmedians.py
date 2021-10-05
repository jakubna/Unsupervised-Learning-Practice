from algorithms.kmeans import Kmeans
import numpy as np


class Kmedians(Kmeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Distance function - Manhattan Distance Function
        self.metric = 'cityblock'

    def _update_centroids(self):
        """Compute the new centroid for each cluster using method depending on algorithm."""
        for k in range(self.M):
            nearest_ids = self.delta[:, k] == 1
            nearests = self.X[nearest_ids]
            if len(nearests) > 0:
                self.centroids[k, :] = np.median(np.array(nearests), axis=0)
