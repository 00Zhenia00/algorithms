import pandas as pd

class OneRuleClassifier:
    def __init__(self):
        self._classification_feature = ""
        self._classification_map = {}
        self._target_mode = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rows_num: int = X.shape[0]
        df: pd.DataFrame = X.copy()
        df["target"] = y  # Ensure y is a Series and correctly added to DataFrame
                
        features: list = X.columns
        target: str = "target"
        
        self._target_mode = y.mode().to_list()[-1]

        min_feature_score: float = 1.1
        classification_map: dict = {}
        classification_feature: str = ""

        for feature in features:
            freq_table = (
                pd.get_dummies(df[[feature, target]], columns=[target], dtype=int, prefix="", prefix_sep="")
                .groupby(feature)
                .sum()
            )
            freq_table = freq_table.assign(MinValue=freq_table.min(axis=1), MaxValueCol=freq_table.idxmax(axis=1))
            cls_map = freq_table["MaxValueCol"].to_dict()
            feature_score = freq_table["MinValue"].sum() / rows_num
            
            if feature_score < min_feature_score:
                classification_feature = feature
                classification_map = cls_map
                min_feature_score = feature_score
        
        self._classification_feature = classification_feature
        self._classification_map = classification_map

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self._classification_feature].map(self._classification_map).fillna(self._target_mode).astype(int)
    
    @property
    def classification_feature(self):
        return self._classification_feature
    
    @property
    def classification_map(self):
        return self._classification_map

