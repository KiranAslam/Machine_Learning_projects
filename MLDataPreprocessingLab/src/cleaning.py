import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def handle_mixed_type(self):
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except ValueError:
                    pass
        return self.df

    def impute_data(self, strategy='knn', columns=None, n_neighbors=5):
        cols_to_fix = columns if columns else self.df.columns.tolist()
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            return self.df, None

        self.df[cols_to_fix] = imputer.fit_transform(self.df[cols_to_fix])
        return self.df, imputer