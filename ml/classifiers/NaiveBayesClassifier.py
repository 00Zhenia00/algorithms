import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self._classes = list()
        self._prior_probs = dict()
        self._conditional_probs = dict()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df: pd.DataFrame = X.copy()
        df["target"] = y  # Ensure y is a Series and correctly added to DataFrame
        features: list = X.columns
        target: str = "target"

        # Laplace smoothing params
        self._unique_values_num = pd.unique(df[features].values.ravel()).shape[0]
        self._class_counts = df[target].value_counts().to_dict()

        # Compute prior probabilities P(S)
        samples_num = sum(self._class_counts.values())
        self._prior_probs = {cls: count/samples_num for cls, count in self._class_counts.items()}

        # Compute available class labels
        self._classes = df[target].unique()

        # Compute conditional probabilities P(Qi | S)
        for feature in features:
            self._conditional_probs[feature] =(
                (df.groupby([target, feature]).size().add(1))
                .div(
                    df.groupby(target).size().add(self._unique_values_num),
                    level=0)
            )

    def predict(self, X: pd.DataFrame):
        features = X.columns
        y = X.apply(lambda x: self._predict(x, features), axis=1)
        return y


    def _predict(self, sample: pd.Series, features: list):
        post_probs = {}

        for cls in self._classes:
            prob = self._prior_probs[cls]

            for feature in features:
                if (cls, sample[feature]) in self._conditional_probs[feature]:
                    prob *= self._conditional_probs[feature][(cls, sample[feature])]
                else:
                    prob *= 1 / (self._class_counts[cls] + self._unique_values_num)
        
            post_probs[cls] = prob

        # Return class with highest probability
        return max(post_probs, key=post_probs.get)
