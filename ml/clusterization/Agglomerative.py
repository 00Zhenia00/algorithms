import numpy as np

class Agglomerative:
    def __init__(
        self, n_clusters=2, threshold=None, metric="euclidean", linkage="single"
    ):
        self._data = None
        self._dist_matrix = None
        self._labels = None
        self._n_clusters = n_clusters
        self._threshold = threshold
        self._metric = metric
        self._linkage = linkage

    def fit(self, data):
        self._data = data
        self._dist_matrix = np.zeros((len(data), len(data)), dtype=float)
        self._labels = np.arange(len(data))

        # Calculate initial distance matrix
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                self._dist_matrix[i, j] = self._distance(data[i], data[j])
                self._dist_matrix[j, i] = self._dist_matrix[i, j]

        if self._threshold is None:
            self._threshold = np.inf

        min_dist = None
        while (len(np.unique(self._labels)) > self._n_clusters) and (
            min_dist is None or min_dist < self._threshold
        ):
            self._dist_matrix, self._labels, min_dist = self._update(
                self._dist_matrix, self._labels
            )

        return self

    def _single_linkage_distance(self, x1, x2):
        return np.min(
            [
                self._distance(x1[i], x2[j])
                for i in range(len(x1))
                for j in range(len(x2))
            ]
        )

    def _complete_linkage_distance(self, x1, x2):
        return np.max(
            [
                self._distance(x1[i], x2[j])
                for i in range(len(x1))
                for j in range(len(x2))
            ]
        )

    def _average_linkage_distance(self, x1, x2):
        return np.mean(
            [
                self._distance(x1[i], x2[j])
                for i in range(len(x1))
                for j in range(len(x2))
            ]
        )

    def _ward_linkage_distance(self, x1, x2):
        return (
            np.sum((np.mean(x1, axis=0) - np.mean(x2, axis=0)) ** 2)
            * len(x1)
            * len(x2)
            / (len(x1) + len(x2))
        )

    def _linkage_distance(self, x1, x2):
        if self._linkage == "single":
            return self._single_linkage_distance(x1, x2)
        elif self._linkage == "complete":
            return self._complete_linkage_distance(x1, x2)
        elif self._linkage == "average":
            return self._average_linkage_distance(x1, x2)
        elif self._linkage == "ward":
            return self._ward_linkage_distance(x1, x2)
        else:
            raise ValueError(f"Unknown linkage: {self._linkage}!")

    def _distance(self, x1, x2):
        if self._metric == "euclidean":
            return np.linalg.norm(x1 - x2)
        else:
            raise ValueError(f"Unknown metric: {self._metric}!")

    def _find_first_occurrence_indices(self, matrix, value):
        mask = matrix == value
        if mask.any():
            return np.unravel_index(mask.argmax(), matrix.shape)
        return (None, None)

    def _update(self, dist_matrix, labels):
        indices_upper = np.triu_indices(
            len(dist_matrix), k=1
        )  # Index of upper part of dist_matrix (skip diagonal)
        min_distance = np.min(dist_matrix[indices_upper])
        row, col = self._find_first_occurrence_indices(dist_matrix, min_distance)

        # Update label
        labels[labels == col] = row
        labels[labels > col] -= 1

        # Deleted the row and column 'col'
        dist_matrix = np.delete(dist_matrix, col, 0)
        dist_matrix = np.delete(dist_matrix, col, 1)

        # Update dist_matrix
        for i in range(len(dist_matrix)):
            dist_matrix[row, i] = self._linkage_distance(
                self._data[labels == row], self._data[labels == i]
            )
            dist_matrix[i, row] = dist_matrix[row, i]
        return dist_matrix, labels, min_distance

    @property
    def labels(self):
        return self._labels
