import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances


class Kmeans:
    def __init__(self, M: int, init: str = 'random', tol=1e-6, t_max: int = 500, seed=-1, centroids=None):
        """
        :param M: Number of Clusters
        :param init: Possible initialization methods: [random, kpp]
        :param t_max: Maximum number of iterations if hasn't reached convergence yet
        :param seed: Fixed seed to allow reproducibility
        :param tol: Relative tolerance for the difference in cluster centers of two consecutive iterations
        that declare convergence
        :param centroids: derived centroids to initialize k-Means run with
        """
        if M < 1:
            raise ValueError('M must be a positive number')
        if init not in ['random', 'kpp']:
            raise ValueError('Wrong init value. Possible initialization methods: [random, kpp]')

        self.M = M
        self.t_max = t_max
        self.seed = seed
        self.tol = tol
        self.metric = 'euclidean'
        self.alg_init_centroids = centroids
        self.previous_centroids = None
        self.centroids = None
        self.t = 0
        self.init = init

        self.ew = 0
        self.emax = 0
        self.esum = 0
        self.V = np.zeros(self.M)
        self.labels_ = None

        self.X = None
        self.delta = None

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
        if self.alg_init_centroids is not None:
            self.centroids = self.alg_init_centroids
        else:
            self._init_centroids()
        while True:
            self.t += 1
            # Update clusters
            self.delta, nearest_ids, self.labels_ = self._update_clusters(x)
            self.previous_centroids = self.centroids.copy()
            self._update_centroids()
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
        if self.init == 'random':
            init_centroids = np.random.choice(range(self.X.shape[0]), size=self.M, replace=False)
            self.centroids = self.X[init_centroids, :]
        elif self.init == "kpp":
            # randomly choose the first centroid
            centroids = np.zeros((self.M, self.X.shape[1]))
            rand_index = np.random.choice(self.X.shape[0])
            centroids[0] = self.X[rand_index]
            # compute distances from the first centroid chosen to all the other data points
            distances = pairwise_distances(self.X, [centroids[0]], metric=self.metric).flatten()
            for i in range(1, self.M):
                # choose the next centroid, the probability for each data point to be chosen
                # is directly proportional to its squared distance from the nearest centroid
                prob = distances ** 2
                rand_index = np.random.choice(self.X.shape[0], size=1, p=prob / np.sum(prob))
                centroids[i] = self.X[rand_index]
                if i == self.M - 1:
                    break

                # if we still need another cluster,
                # compute distances from the centroids to all data points
                # and update the squared distance as the minimum distance to all centroid
                distances_new = pairwise_distances(self.X, [centroids[i]], metric=self.metric).flatten()
                distances = np.min(np.vstack((distances, distances_new)), axis=0)
            self.centroids = centroids

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
        delta = np.zeros(shape=(x.shape[0], self.M))
        clusters = []
        for row_id, row in enumerate(x):
            for centroid_id, centroid in enumerate(self.centroids):
                d = self._calculate_distance(centroid, row)
                distances[row_id, centroid_id] = d
            cluster_id = int(np.argmin(distances[row_id, :]))
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
        """Update the values of Emax and Esum"""
        self.esum = np.sum(self.V)
        self.emax = np.max(self.V)

    def _calculate_sse(self):
        """
        Calculate the distance between old centroids and new ones (SSE).
        :return: SSE.
        """
        cost = 0
        for k in range(self.M):
            cost += \
                distance.cdist(np.array([self.centroids[k]]), np.array([self.previous_centroids[k]]),
                               metric=self.metric)[0][0]

        V_t = np.zeros(self.M)
        for row_id, row in enumerate(self.X):
            for centroid_id, centroid in enumerate(self.centroids):
                if self.delta[row_id, centroid_id] == 1:
                    V_t[centroid_id] += self._calculate_distance(centroid, row)
        self.V = V_t
        ew_t_1 = self.ew
        self.ew = np.sum(V_t)
        self.emax = np.max(V_t)
        self.esum = np.sum(V_t)
        self._update_ew()
        return cost

    def _check_convergence(self):
        """Check the termination criterion"""
        if self.t >= self.t_max:
            return True
        elif self._calculate_sse() < self.tol:
            return True
        else:
            return False
