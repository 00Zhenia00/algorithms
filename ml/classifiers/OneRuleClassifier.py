import pandas as pd

class OneRuleClassifier:
    def __init__(self):
        self._classification_feature = str()
        self._classification_map = dict()

        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        rows_num: int = X.shape[0]
        df: pd.DataFrame = pd.concat([X, y], axis=1)
        features: list = df.columns[:-1]
        target: str = df.columns[-1]

        min_feature_score: float = 1.1
        classification_map: dict = dict()
        classification_feature: dict = str()

        for feature in features:
            freq_table = pd.get_dummies(df[[feature, target]], columns=[target], dtype=int, prefix="", prefix_sep="").groupby(feature).sum()
            freq_table = freq_table.assign(MinValue = freq_table.min(axis=1), MaxValueCol = freq_table.idxmax(axis=1))
            cls_map = freq_table["MaxValueCol"].to_dict()
            feature_score = float(freq_table[["MinValue"]].sum().iloc[0]) / rows_num
            if feature_score < min_feature_score:
                classification_feature = feature
                classification_map = cls_map
                min_feature_score = feature_score
        
        self._classification_feature = classification_feature
        self._classification_map = classification_map

    def predict(self, X: pd.DataFrame):
        y: pd.Series = X.loc[:, [self._classification_feature]].map(lambda x: self._classification_map[x])[:, -1]
        return y
