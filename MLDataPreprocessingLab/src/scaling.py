import pandas as pd
# fixed: RobustScaler was used but never imported
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataScaler:
    def __init__(self, df):
        self.df = df.copy()

    def scale_data(self, columns=None, method='standard'):
        cols_to_scale = columns if columns else self.df.select_dtypes(include=['number']).columns.tolist()
        if not cols_to_scale:
            return self.df, None

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return self.df, None

        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        return self.df, scaler