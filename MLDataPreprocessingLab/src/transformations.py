import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

class DataTransformer:
    def __init__(self, df):
        self.df = df.copy()

    def apply_log_transform(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])
        return self.df

    def apply_power_transform(self, columns, method='yeo-johnson'):
        if not columns:
            return self.df
        pt = PowerTransformer(method=method)
        self.df[columns] = pt.fit_transform(self.df[columns])
        return self.df, pt