import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierHandler:
    def __init__(self, df):
        self.df = df.copy()

    def detect_zscore(self, column, threshold=3):
        z_score = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        return z_score > threshold

    def detect_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

    def detect_isolation_forest(self, column, contamination=0.05):
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(self.df[[column]])
        return preds == -1

    def handle_outliers(self, column, method='iqr', action='cap'):
        if method == 'iqr':
            is_outlier = self.detect_iqr(column)
        elif method == 'zscore':
            is_outlier = self.detect_zscore(column)
        elif method == 'iso_forest':
            is_outlier = self.detect_isolation_forest(column)
        else:
            return self.df

        if action == 'remove':
            self.df = self.df[~is_outlier]

        elif action == 'cap':
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            else:
                mean = self.df[column].mean()
                std = self.df[column].std()
                lower = mean - (3 * std)
                upper = mean + (3 * std)
            self.df[column] = self.df[column].clip(lower, upper)

        return self.df