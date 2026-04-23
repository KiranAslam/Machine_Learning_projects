import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

class DataBalancer:
    def __init__(self, df):
        self.df = df.copy()

    def balance_data(self, target_col, method='smote'):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            return self.df

        X_res, y_res = sampler.fit_resample(X, y)
        balanced_df = pd.concat([
            pd.DataFrame(X_res, columns=X.columns),
            pd.DataFrame(y_res, columns=[target_col])
        ], axis=1)
        return balanced_df