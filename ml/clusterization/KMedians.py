import numpy as np

class KMedians:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init="random", random_state=None):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._init = init
        self._data = None
        self._labels = None
        self._n_samples = None
        self._n_features = None
        self._centroids = None
        np.random.seed(random_state)
        self._check_params()

    def _check_params(self):
        self._check_input_types()
        self._check_input_values()

    def _check_input_types(self):
        if not isinstance(self._n_clusters, int):
            raise TypeError("`n_clusters` parameter must be an integer.")
        if not isinstance(self._max_iter, int):
            raise TypeError("`max_iter` parameter must be an integer.")
        if not isinstance(self._tol, (int, float)):
            raise TypeError("`tol` parameter must be a number.")
        if not isinstance(self._init, str):
            raise TypeError("`init` parameter must be a string.")
    
    def _check_input_values(self):
        if self._n_clusters <= 0:
            raise ValueError("`n_clusters` parameter must be a positive integer.")
        if self._max_iter <= 0:
            raise ValueError("`max_iter` parameter must be a positive integer.")
        if self._tol < 0:
            raise ValueError("`tol` parameter must be a non-negative number.")
        if self._init not in ["random", "k-medians++"]:
            raise ValueError("`init` parameter must be 'random' or 'k-medians++'.")
    
    def _init_clusters_random(self):
        random_indices = np.random.choice(self._n_samples, self._n_clusters, replace=False)
        self._centroids = self._data[random_indices]

    def _init_clusters_plusplus(self):
        self._centroids = np.empty((self._n_clusters, self._n_features))
        self._centroids[0] = self._data[np.random.choice(self._n_samples)]

        for i in range(1, self._n_clusters):
            distances = np.linalg.norm(self._data[:, np.newaxis] - self._centroids[:i], axis=2)
            probabilities = distances / distances.sum()
            self._centroids[i] = self._data[np.random.choice(self._n_samples, p=probabilities)]

    def _init_clusters(self):
        if self._init == "random":
            self._init_clusters_random()
        elif self._init == "k-medians++":
            self._init_clusters_plusplus()
    
    def _check_input_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D array.")
        if data.shape[0] < self._n_clusters:
            raise ValueError("Number of samples must be greater than or equal to n_clusters.")

    def fit(self, data):
        self._check_input_data(data)
        self._data = data
        self._n_samples, self._n_features = data.shape
        self._init_clusters()

        for _ in range(self._max_iter):
            # Assign clusters
            distances = np.linalg.norm(data[:, np.newaxis] - self._centroids, axis=2)
            self._labels = np.argmin(distances, axis=1)
            # Update centroids
            new_centroids = np.array([np.median(data[self._labels == i], axis=0) for i in range(self._n_clusters)])
            # Check for convergence
            if np.all(np.abs(new_centroids - self._centroids) < self._tol):
                break
            self._centroids = new_centroids
        return self
    
    @property
    def labels(self):
        return self._labels