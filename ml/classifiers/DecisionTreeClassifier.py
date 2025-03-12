import pandas as pd
import numpy as np

def entropy(series: pd.Series) -> float:
    """Calculate the entropy of a Pandas Series."""
    value_counts = series.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-9))  # Avoid log(0)

def information_gain(X: pd.DataFrame, y: pd.Series, split_col: str) -> float:
    """
    Calculate the Information Gain of splitting `y` based on `split_col` in `X`.

    Parameters:
    X : pd.DataFrame - The feature dataset.
    y : pd.Series - The target variable.
    split_col : str - The column in X to split on.

    Returns:
    float - The information gain.
    """
    total_entropy = entropy(y)

    split_values = X[split_col].unique()
    weighted_entropy = 0

    for val in split_values:
        y_sub = y[X[split_col] == val]
        prob = len(y_sub) / len(y)
        weighted_entropy += prob * entropy(y_sub)

    return total_entropy - weighted_entropy

class Node:
    def __init__(self, split_feature: str = None, split_dict: dict = None, class_label: str = None):
        self._split_feature: str = split_feature
        self._split_dict: dict = split_dict
        self._class_label: str = class_label
    
    @property
    def class_label(self):
        return self._class_label
    
    @property
    def split_dict(self):
        return self._split_dict

    @property
    def split_feature(self):
        return self._split_feature
    
    def is_leaf(self) -> bool:
        return self._class_label is not None


class DecisionTreeClassifier:
    def __init__(self, depth=None, logging=False):
        self._head_node = None
        self._depth = depth
        self._logging = logging

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._log("fit(): Start fit...")
        self._head_node = self._build_tree(X, y, X.columns.to_list(), self._depth)
        self._log("fit(): Finish fit.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return np.array([self._search_tree(sample, self._head_node) for _, sample in X.iterrows()])
    
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, features: list, depth: int = None) -> Node:
        if len(features) == 0:
            return Node(class_label=y.mode().to_list()[-1])
        
        if depth:
            if depth > self._depth:
                return Node(class_label=y.mode().to_list()[-1])

        if y.nunique() == 1:
            return Node(class_label=pd.unique(y)[-1])
            
        max_information_gain = -1
        max_IG_feature = str()
        for feature in features:
            ig = information_gain(X, y, feature)
            if ig > max_information_gain:
                max_information_gain = ig
                max_IG_feature = feature

        split_values = pd.unique(X[max_IG_feature])

        if len(split_values) == 1:
            return Node(class_label=pd.unique(y)[-1])

        self._log(f"_build_tree(): Split feature:{max_IG_feature} | IG:{max_information_gain}")

        child_nodes = dict()
        for value in split_values:
            mask = X[max_IG_feature] == value
            child_nodes[str(value)] = self._build_tree(X[mask], y[mask], list(set(features)-{max_IG_feature}), depth + 1 if depth else None)
        
        return Node(split_feature=max_IG_feature, split_dict=child_nodes)

    def _search_tree(self, sample: pd.Series, node: Node) -> int:
        if node.is_leaf():
            return node.class_label
        else:
            return self._search_tree(sample, node.split_dict[str(sample[node.split_feature])])
    
    def _log(self, s: str):
        if self._logging:
            print(f"[LOG] DecisionTreeClassifier: {s}")


if __name__ == "__main__":
    data = [
        (0, 0, 1, 0, 0),
        (1, 0, 0, 1, 1),
        (2, 0, 1, 0, 1),
        (0, 1, 0, 0, 1),
        (0, 1, 1, 0, 1),
        (0, 1, 1, 1, 0),
        (1, 0, 0, 1, 0),
        (2, 0, 0, 0, 1),
        (2, 1, 1, 0, 1),
        (0, 1, 1, 1, 0)
    ]

    # Creating DataFrame with specified column names
    df = pd.DataFrame(data, columns=['Q1', 'Q2', 'Q3', 'Q4', 'S'])

    X = df.drop(["S"], axis=1)
    y = df["S"]

    model = DecisionTreeClassifier(logging=True)
    model.fit(X, y)

    X_inf = pd.DataFrame([(2, 1, 1, 1)],columns=['Q1', 'Q2', 'Q3', 'Q4'])
    y_pred = model.predict(X_inf)

    df_result = X_inf.copy()
    df_result["prediction"] = y_pred
    print(df_result)
