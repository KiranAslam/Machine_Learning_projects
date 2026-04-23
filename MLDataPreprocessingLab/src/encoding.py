import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce


class DataEncoder:
    def __init__(self, df):
        self.df = df.copy()

    def one_hot_encode(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        if not columns:
            return self.df, None
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return self.df, columns

    def label_encode(self, columns):
        le = LabelEncoder()
        for col in columns:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        return self.df, le

    def ordinal_encode(self, columns, categories_order=None):
        if not columns:
            return self.df, None
        oe = OrdinalEncoder(categories=categories_order)
        self.df[columns] = oe.fit_transform(self.df[columns])
        return self.df, oe

    def binary_encode(self, columns):
        if not columns:
            return self.df, None
        encoder = ce.BinaryEncoder(cols=columns)
        self.df = encoder.fit_transform(self.df)
        return self.df, encoder

    def frequency_encode(self, columns):
        freq_maps = {}
        for col in columns:
            if col in self.df.columns:
                freq = self.df[col].value_counts()
                self.df[col] = self.df[col].map(freq)
                freq_maps[col] = freq
        return self.df, freq_maps
        
    def target_encode(self, columns=None, column=None, target_column=None, smoothing=10):
        col = column if column else (columns[0] if columns else None)
        if col is None or target_column not in self.df.columns:
            return self.df, None

        global_mean = self.df[target_column].mean()
        stats = self.df.groupby(col)[target_column].agg(['count', 'mean'])
        smooth_values = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        self.df[col] = self.df[col].map(smooth_values).fillna(global_mean)
        return self.df, (smooth_values, global_mean)