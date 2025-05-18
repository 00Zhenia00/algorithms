import numpy as np

# pseudo-code for DBSCAN

# DBSCAN(DB, distFunc, eps, minPts) {
#    C=0                                                  /* Счётчик кластеров */
#    for each point P in database DB {
#       if label(P) ≠ undefined then continue               /* Точка была просмотрена во внутреннем цикле */
#       Neighbors N=RangeQuery(DB, distFunc, P, eps)      /* Находим соседей */
#       if|N|< minPts then {                              /* Проверка плотности */
#          label(P)=Noise                                 /* Помечаем как шум */
#          continue
#       }
#       C=C + 1                                           /* следующая метка кластера */
#       label(P)=C                                        /* Помечаем начальную точку */
#       Seed set S=N \ {P}                                /* Соседи для расширения */
#       for each point Q in S {                             /* Обрабатываем каждую зачаточную точку */
#          if label(Q)=Noise then label(Q)=C            /* Заменяем метку Шум на Край */
#          if label(Q) ≠ undefined then continue            /* Была просмотрена */
#          label(Q)=C                                     /* Помечаем соседа */
#          Neighbors N=RangeQuery(DB, distFunc, Q, eps)   /* Находим соседей */
#          if|N|≥ minPts then {                           /* Проверяем плотность */
#             S=S ∪ N                                     /* Добавляем соседей в набор зачаточных точек */
#          }
#       }
#    }
# }

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self._eps = eps
        self._min_samples = min_samples
        self._labels = None
        self._data = None
        self._check_params()

    def _check_params(self):
        self._check_types()
        self._check_values()

    def _check_types(self):
        if not isinstance(self._eps, (int, float)):
            raise TypeError("eps must be a number")
        if not isinstance(self._min_samples, int):
            raise TypeError("min_samples must be an integer")
        
    def _check_values(self):
        if self._eps <= 0:
            raise ValueError("eps must be greater than 0")
        if self._min_samples <= 0:
            raise ValueError("min_samples must be greater than 0")

    def fit(self, data):
        self._data = data
        self._labels = np.full(data.shape[0], -1)
        cluster_id = 0

        for point_idx in range(data.shape[0]):
            if self._labels[point_idx] != -1:
                continue

            neighbors = self._region_query(point_idx)

            if len(neighbors) < self._min_samples:
                self._labels[point_idx] = -1  # Noise
                continue

            self._labels[point_idx] = cluster_id
            self._expand_cluster(neighbors, cluster_id)

            cluster_id += 1
        return self

    def _region_query(self, point_idx):
        """
        Find all points within eps distance from the point at point_idx.
        """
        neighbors = []
        for i, point in enumerate(self._data):
            if np.linalg.norm(point - self._data[point_idx]) < self._eps:
                neighbors.append(i)
        return neighbors
    
    def _expand_cluster(self, neighbors, cluster_id):
        """
        Expand the cluster by checking the neighbors.
        """
        for neighbor_idx in neighbors:
            if self._labels[neighbor_idx] == -1:
                self._labels[neighbor_idx] = cluster_id
            elif self._labels[neighbor_idx] != -1:
                continue

            self._labels[neighbor_idx] = cluster_id
            new_neighbors = self._region_query(neighbor_idx)

            if len(new_neighbors) >= self._min_samples:
                neighbors.extend(new_neighbors)
        return neighbors
    
    @property
    def labels(self):
        return self._labels
