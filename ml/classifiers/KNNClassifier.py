import pandas as pd
import numpy as np

def euclidean_distance(x1: pd.Series, x2: pd.Series):
    """
    Computes the Euclidean distance between two pandas Series.

    Parameters:
    x1 (pd.Series): First data point.
    x2 (pd.Series): Second data point.

    Returns:
    float: The Euclidean distance between x1 and x2.
    """
    return np.sqrt(((x1 - x2) ** 2).sum())

class KNNClassifier:
    def __init__(self, k=3):
        self._X_train = None
        self._y_train = None
        self._k = k

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._X_train = X
        self._y_train = y

    def predict(self, X: pd.DataFrame):
        y_pred = [self._predict_sample(x) for _, x in X.iterrows()]
        return np.array(y_pred)

    def _predict_sample(self, sample: pd.Series):
        distances = [euclidean_distance(sample, train_sample) for _, train_sample in self._X_train.iterrows()]
        indices = np.argsort(distances)[: self._k]
        return self._y_train[indices].mode().to_list()[-1]
