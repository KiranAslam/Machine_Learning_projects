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
        cols_to_fix = [c for c in cols_to_fix if c in self.df.columns]

        if not cols_to_fix:
            return self.df, None

        is_single_column = len(cols_to_fix) == 1
        is_string_col = all(
            not pd.api.types.is_numeric_dtype(self.df[c]) for c in cols_to_fix
        )

        if is_string_col:
            imputer = SimpleImputer(strategy='most_frequent')
            cols_with_nulls = [c for c in cols_to_fix if self.df[c].isnull().any()]
            if not cols_with_nulls:
                return self.df, None
            self.df[cols_with_nulls] = imputer.fit_transform(self.df[cols_with_nulls])
            return self.df, imputer
        if strategy == 'knn' and is_single_column:
            strategy = 'mean'

        if strategy == 'iterative' and is_single_column:
            strategy = 'median'

        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            return self.df, None
        numeric_cols_with_nulls = [
            c for c in cols_to_fix
            if pd.api.types.is_numeric_dtype(self.df[c]) and self.df[c].isnull().any()
        ]
        if not numeric_cols_with_nulls:
            return self.df, None
        self.df[numeric_cols_with_nulls] = imputer.fit_transform(self.df[numeric_cols_with_nulls])
        return self.df, imputer