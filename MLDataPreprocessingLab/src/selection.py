import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, RFE, SelectKBest,
    mutual_info_classif, mutual_info_regression, chi2
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV

class FeatureSelector:
    def __init__(self, df):
        self.df = df.copy()

    def remove_low_variance(self, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return self.df, []
        selector.fit(self.df[numeric_cols])
        kept_cols = numeric_cols[selector.get_support()]
        dropped = list(set(numeric_cols) - set(kept_cols))
        self.df = self.df.drop(columns=dropped)
        return self.df, dropped

    def correlation_filter(self, threshold=0.9):
        numeric_df = self.df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.df = self.df.drop(columns=to_drop)
        return self.df

    # fixed: was always using RandomForestClassifier, now accepts task param to support regression
    def rfe_selection(self, target_col, n_features=5, task='classification'):
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = self.df[target_col]
        estimator = (
            RandomForestClassifier(n_estimators=50, random_state=42)
            if task == 'classification'
            else RandomForestRegressor(n_estimators=50, random_state=42)
        )
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        selected_cols = X.columns[selector.support_].tolist()
        return self.df[selected_cols + [target_col]]

    def mutual_info_selection(self, target_col, k=5, task='classification'):
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = self.df[target_col]
        score_func = mutual_info_classif if task == 'classification' else mutual_info_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        return self.df[selected_cols + [target_col]]

    def lasso_selection(self, target_col):
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = self.df[target_col]
        lasso = LassoCV(cv=5).fit(X, y)
        selected_cols = X.columns[np.abs(lasso.coef_) > 0].tolist()
        if not selected_cols:
            selected_cols = X.columns[np.argsort(np.abs(lasso.coef_))[-5:]].tolist()
        return self.df[selected_cols + [target_col]]

    def chi2_selection(self, target_col, k=5):
        X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
        X_pos = X.apply(lambda x: x + abs(x.min()) if x.min() < 0 else x)
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X_pos, self.df[target_col])
        selected_cols = X.columns[selector.get_support()].tolist()
        return self.df[selected_cols + [target_col]]