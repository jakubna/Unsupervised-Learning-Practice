import numpy as np
from scipy.spatial import distance


class MinMaxKmeans:
    def __init__(self, M: int, p_max=0.5, p_step=0.01, beta=0.1, tol=1e-6, t_max: int = 500, seed=-1):
        """
        :param M: Number of Clusters
        :param p_max: Maximum value of p exponent from the range [0,1)
        :param p_step: The value by which p exponent is increased after each iteration
        :param beta: Parameter controlling the influence of the previous weight to the current update;
        belongs to the range [0,1]
        :param t_max: Maximum number of iterations if hasn't reached convergence yet
        :param seed: Fixed seed to allow reproducibility
        :param tol: Relative tolerance for the difference in cluster centers of two consecutive iterations
        that declare convergence
        """
        if M < 1:
            raise ValueError('M must be a positive number')
        if p_max < 0 or p_max >= 1:
            raise ValueError('p must belong to the set [0,1)')
        if beta < 0 or beta > 1:
            raise ValueError('beta must belong to the set [0,1]')

        self.M = M
        self.p_max = p_max
        self.p_step = p_step
        self.beta = beta
        self.t_max = t_max
        self.seed = seed
        self.tol = tol
        self.metric = 'euclidean'

        self.centroids = None
        self.t = 0
        self.p_init = 0
        self.p = 0

        self.w = np.ones(self.M) * 1 / self.M
        self.w_p = {self.p: np.ones(self.M) * 1 / self.M}
        self.delta = None
        self.delta_p = {}
        self.ew = 0
        self.emax = 0
        self.esum = 0
        self.V = np.zeros(self.M)
        self.labels_ = None
        self.X = None

    def fit(self, x: np.ndarray):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param x: 2D data array of size (rows, features).
        """
        if self.seed < 0:
            np.random.seed()
        else:
            np.random.seed(self.seed)
        self.X = x
        # Initialize centroids
        self._init_centroids()
        # No empty or singleton clusters yet detected
        empty = False
        while True:
            self.t += 1
            self.delta, nearest_ids, self.labels_ = self._update_clusters(x)
            # If empty or singleton clusters have occurred at time t then reduce p.
            if False in [len(i) > 1 for i in nearest_ids]:
                empty = True
                self.p = np.round((self.p - self.p_step), 2)
                if self.p < self.p_init:
                    return None
                # Revert to the assignments and weights corresponding to the reduced p
                self.w = self.w_p[self.p]
                self.delta = self.delta_p[self.p]
            self._update_centroids()
            if self.p < self.p_max and empty == False:
                # Store the current assignments corresponding to p
                self.delta_p[self.p] = self.delta
                # Store the weights corresponding to p
                self.w_p[self.p] = self.w
                # Increase the value of p byvariance  p_step
                self.p = np.round((self.p + self.p_step), 2)
            # Update the weights
            self._update_weights()
            # Check the termination criterion
            if self._check_convergence():
                break

    def predict(self, x: np.ndarray):
        """
        Assign labels to a list of observations.
        :param x: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        if self.centroids is None:
            raise Exception('Fit the model with some data before running a prediction')

        distances = self._distances(x)
        labels, nearest, nearest_ids = self._get_nearest(x, distances)

        return labels

    def fit_predict(self, x_train: np.ndarray, x_test: np.ndarray):
        """
        Fit the model with data and return assigned labels.
        :param x_train: 2D data array of size (rows, features).
        :param x_test: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        self.fit(x_train)
        return self.predict(x_test)

    def _init_centroids(self):
        """Initialize centroids"""
        init_centroids = np.random.choice(range(self.X.shape[0]), size=self.M, replace=False)
        self.centroids = self.X[init_centroids, :]
    def _distances(self, x: np.ndarray):
        """
        Calculate distance from each point of the dataset to each cluster.
        :param x: 2D data array of size (rows, features).
        :return: Distance matrix of shape (M, number of points)
        """
        distances = np.zeros(shape=(x.shape[0], self.M))
        for row_id, row in enumerate(x):
            for centroid_id, centroid in enumerate(self.centroids):
                distances[row_id, centroid_id] = self._calculate_distance(centroid, row)
        return distances

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
        weighted_distances = np.zeros(shape=(x.shape[0], self.M))
        delta = np.zeros(shape=(x.shape[0], self.M))
        clusters = []
        for row_id, row in enumerate(x):
            for centroid_id, centroid in enumerate(self.centroids):
                d = self._calculate_distance(centroid, row)
                distances[row_id, centroid_id] = d
                weighted_distances[row_id, centroid_id] = np.power(self.w[centroid_id], self.p) * d
            cluster_id = int(np.argmin(weighted_distances[row_id, :]))
            clusters.append(cluster_id)
            nearest_id[cluster_id].append(row_id)
            delta[row_id, cluster_id] = 1
        return delta, nearest_id, clusters

    def _calculate_distance(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate distance between 2 elements using the metric depending on the algorithm ('euclidean').
        :param x: 1D vector with all x attributes.
        :param y: 1D vector with all y attributes.
        :return: Distance between both vectors using the specified metric.
        """
        return distance.cdist(np.array([x]), np.array([y]), metric=self.metric)[0][0]

    def _update_centroids(self):
        """Compute the new centroid for each cluster."""
        for k in range(self.M):
            nearest_ids = self.delta[:, k] == 1
            nearests = self.X[nearest_ids]
            self.centroids[k, :] = np.mean(np.array(nearests), axis=0)

    def _update_weights(self):
        """Update the weights for each cluster."""
        V_t = np.zeros(self.M)
        for row_id, row in enumerate(self.X):
            for centroid_id, centroid in enumerate(self.centroids):
                if self.delta[row_id, centroid_id] == 1:
                    V_t[centroid_id] += self._calculate_distance(centroid, row)
        self.V = V_t
        z = np.sum(np.power(V_t, 1 / (1 - self.p)))
        w_t = np.zeros(self.M)
        for k in range(self.M):
            w_t[k] = self.beta * self.w[k] + (1 - self.beta) * (np.power(V_t[k], 1 / (1 - self.p)) / z)
        self.w = w_t

    def _get_nearest(self, x: np.ndarray, distances: np.ndarray):
        """
        Compute the distance for each dataset instance to each centroid.
        :param x: 2D data array of size (rows, features).
        :param distances: 2D vector of distances between centroids and points.
        :return clusters: Cluster indexes assigned to each observation (labels for each point).
        :return nearest: List of nearest observations for each cluster.
        :return nearest_id: List of nearest observations index for each cluster.
        """
        clusters = []
        nearest = [[] for _ in range(self.M)]
        nearest_id = [[] for _ in range(self.M)]

        for row_id, row in enumerate(x):
            cluster_id = int(np.argmin(distances[row_id, :]))

            clusters.append(cluster_id)
            nearest[cluster_id].append(row)
            nearest_id[cluster_id].append(row_id)

        return clusters, nearest, nearest_id

    def _update_ew(self):
        """Compute the values of Emax and Esum"""
        self.esum = np.sum(self.V)
        self.emax = np.max(self.V)

    def _ew_difference(self):
        """
        Calculate the difference between Ew(t) and Ew(t-1);
        Ew - a weighted formulation of the sum of the intra-clusters variances
        :return: |Ew(t) - Ew(t-1)|.
        """
        ew_t_1 = self.ew
        ew_t = 0
        for k in range(self.M):
            ew_t += np.power(self.w[k], self.p) * self.V[k]
        self.ew = ew_t
        self._update_ew()
        return np.abs(ew_t - ew_t_1)

    def _check_convergence(self):
        """Check the termination criterion"""
        if self.t >= self.t_max:
            return True
        elif self._ew_difference() < self.tol:
            return True
        else:
            return False
