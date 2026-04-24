import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
import category_encoders as ce


class DataEncoder:
    def __init__(self, df):
        self.df = df.copy()

    def one_hot_encode(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return self.df, None
        enc = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded = enc.fit_transform(self.df[columns])
        feature_names = enc.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=self.df.index)
        self.df = self.df.drop(columns=columns)
        self.df = pd.concat([self.df, encoded_df], axis=1)
        return self.df, enc

    def label_encode(self, columns):
        columns = [c for c in columns if c in self.df.columns]
        fitted_encoders = {}
        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            fitted_encoders[col] = le
        return self.df, fitted_encoders

    def ordinal_encode(self, columns, categories_order=None):
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return self.df, None
        oe = OrdinalEncoder(
            categories=categories_order if categories_order else 'auto',
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.df[columns] = oe.fit_transform(self.df[columns])
        return self.df, oe

    def binary_encode(self, columns):
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return self.df, None
        encoder = ce.BinaryEncoder(cols=columns)
        self.df = encoder.fit_transform(self.df)
        return self.df, encoder

    def frequency_encode(self, columns):
        columns = [c for c in columns if c in self.df.columns]
        freq_maps = {}
        for col in columns:
            freq = self.df[col].value_counts()
            self.df[col] = self.df[col].map(freq)
            freq_maps[col] = freq
        return self.df, freq_maps

    def target_encode(self, columns=None, column=None, target_column=None, smoothing=10):
        col = column if column else (columns[0] if columns else None)
        if col is None or col not in self.df.columns:
            return self.df, None
        if target_column not in self.df.columns:
            return self.df, None

        global_mean = self.df[target_column].mean()
        stats = self.df.groupby(col)[target_column].agg(['count', 'mean'])
        smooth_values = (
            (stats['count'] * stats['mean'] + smoothing * global_mean)
            / (stats['count'] + smoothing)
        )
        self.df[col] = self.df[col].map(smooth_values).fillna(global_mean)
        return self.df, (smooth_values, global_mean)