import time

import numpy as np
from sklearn import datasets
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedians import kmedians as pycKMedians


class PyClusteringKMeadiansWrapper:
    """
    Wrapper for pyclustering.cluster.kmedians to use with sklearn-like interface.
    """
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._model = None
        self._labels = None
        self._data = None

    def fit(self, data):
        self._data = data
        [initial_medians, indexes] = kmeans_plusplus(
            self._data, n_clusters=self._n_clusters, random_state=self._random_state
        )
        self._model = pycKMedians(
            self._data,
            initial_medians=initial_medians,
            itermax=self._max_iter,
            tolerance=self._tol,
        )
        self._model.process()
        self._labels = self._cluster_lists_to_labels(self._model.get_clusters())
        return self

    def _cluster_lists_to_labels(self, cluster_lists):
        labels = [0] * len(self._data)
        for i, cluster_list in enumerate(cluster_lists):
            for index in cluster_list:
                labels[index] = i
        return np.array(labels)

    @property
    def labels(self):
        return self._labels


def generate_data(n_samples=1500, random_state=170):
    """
    Generate a variety of datasets for clustering.

    Authors: The scikit-learn developers
    SPDX-License-Identifier: BSD-3-Clause
    """
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state
    )
    noisy_moons = datasets.make_moons(
        n_samples=n_samples, noise=0.05, random_state=random_state
    )
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    rng = np.random.RandomState(random_state)
    no_structure = rng.rand(n_samples, 2), None

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # Blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    
    return [
        ("noisy_circles", noisy_circles),
        ("noisy_moons", noisy_moons),
        ("blobs", blobs),
        ("no_structure", no_structure),
        ("aniso", aniso),
        ("varied", varied),
    ]

def compare_model_plots(models, datasets, params):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(24, 8))

    for row, (model_name, model_fn) in enumerate(models):
        for col, (dataset_name, dataset) in enumerate(datasets):
            X, y = dataset
            X = StandardScaler().fit_transform(X)
            model = model_fn(params[dataset_name])
            t0 = time.time()
            model.fit(X)
            t1 = time.time()
            if hasattr(model, "labels_"):
                labels = model.labels_.astype(int)
            elif hasattr(model, "labels"):
                labels = model.labels
            else:
                labels = model.predict(X)
            ax = axes[row, col]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10")
            ax.set_title(f"{model_name}\n{dataset_name}\nTime: {t1-t0:.2f}s", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(model_name, fontsize=12)
                
    plt.tight_layout()
    plt.show()

def plot_models_per_dataset(models, datasets_list, params):
    for dataset_name, dataset in datasets_list:

        X, y = dataset

        fig, axes = plt.subplots(1, len(models), figsize=(8, 4))

        labels_list = []
        times = []
        sil_scores = []

        for i, (model_name, model_fn) in enumerate(models):
            model = model_fn(params[dataset_name])
            t0 = time.time()
            model.fit(X)
            t1 = time.time()
            if hasattr(model, "labels_"):
                labels = model.labels_
            elif hasattr(model, "labels"):
                labels = model.labels
            else:
                labels = model.predict(X)
            labels_list.append(labels)
            times.append(t1 - t0)
            # Silhouette score only if more than 1 cluster and less than n_samples clusters
            if len(set(labels)) > 1 and len(set(labels)) < len(X):
                sil = silhouette_score(X, labels)
            else:
                sil = float('nan')
            sil_scores.append(sil)
            axes[i].scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10")
            axes[i].set_title(
                f"{model_name}\nTime: {t1-t0:.2f}s, Silhouette: {sil:.2f}",
                fontsize=10
            )
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        # Compute ARI between the two models
        if len(labels_list) == 2:
            ari = adjusted_rand_score(labels_list[0], labels_list[1])
        else:
            ari = float('nan')
        fig.suptitle(f"{dataset_name} | ARI: {ari:.2f}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
