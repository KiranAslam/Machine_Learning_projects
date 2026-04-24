import pandas as pd
import numpy as np
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

        # string/categorical columns can only use most_frequent — remap other strategies
        is_string_col = all(
            not pd.api.types.is_numeric_dtype(self.df[c]) for c in cols_to_fix
        )

        if is_string_col:
            # mean/median/knn/iterative are meaningless on strings — always use most_frequent
            imputer = SimpleImputer(strategy='most_frequent')
            cols_with_nulls = [c for c in cols_to_fix if self.df[c].isnull().any()]
            if not cols_with_nulls:
                return self.df, None
            self.df[cols_with_nulls] = imputer.fit_transform(self.df[cols_with_nulls])
            return self.df, imputer

        # for numeric columns — apply single-column fallbacks where needed
        if strategy == 'knn' and is_single_column:
            # KNN needs multiple columns to compute distances — falls back to mean
            strategy = 'mean'

        if strategy == 'iterative' and is_single_column:
            # IterativeImputer models each feature using all others — needs >1 column
            strategy = 'median'

        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
        elif strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            return self.df, None

        # only run on numeric columns that actually have nulls
        numeric_cols_with_nulls = [
            c for c in cols_to_fix
            if pd.api.types.is_numeric_dtype(self.df[c]) and self.df[c].isnull().any()
        ]
        if not numeric_cols_with_nulls:
            return self.df, None

        self.df[numeric_cols_with_nulls] = imputer.fit_transform(self.df[numeric_cols_with_nulls])

        # returning fitted imputer so processor applies same transform to test without refitting
        return self.df, imputer