import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
    def get_baisc_stats(self):
        stats ={
            'rows' : self.df.shape[0],
            'columns':self.df.shape[1],
            'missing_values':self.df.isnull().sum().sum(),
            'duplicates':self.df.duplicated().sum()
        }
        return stats

    def get_missing_report(self):
        missing_data = self.df.isnull().sum()
        report = missing_data[missing_data >0].reset_index()
        report.columns = ['Column', 'Missing Values']
        report['Percentage'] = (report['Missing Values'] / self.df.shape[0]) * 100
        return report

    def imbalance_report(self, target_column):
        if target_column and target_column in self.df.columns:
            counts = self.df[target_column].value_counts(normalize=True) * 100
            is_imbalanced = counts.min() < 20  
            return counts.to_dict(), is_imbalanced
        return None, False

    def get_columns_type(self):
        numric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object','category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        return {"numeric": numric_cols, "categorical": categorical_cols, 'datetime': datetime_cols}
