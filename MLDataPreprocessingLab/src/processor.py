import pandas as pd
import numpy as np
from src.cleaning import DataCleaner
from src.encoding import DataEncoder
from src.outliers import OutlierHandler
from src.balancing import DataBalancer
from src.selection import FeatureSelector
from src.scaling import DataScaler
from src.transformations import DataTransformer
from sklearn.model_selection import train_test_split

VALID_TRANSFORMS = ['log', 'power']

class DataProcessor:
    def __init__(self):
        self.processed_df = None
        self.train_df = None
        self.test_df = None
        self.logs = []

    def Pipeline(self, df, config):
        self.logs = []
        if not config:
            self.logs.append("No processing steps selected. Returning raw data.")
            return df, self.logs

        self.processed_df = df.copy()
        try:
            if config.get('remove_duplicates'):
                dup_count = self.processed_df.duplicated().sum()
                self.processed_df = self.processed_df.drop_duplicates()
                self.logs.append(f"Removed {dup_count} duplicate rows.")

            cleaner = DataCleaner(self.processed_df)
            self.processed_df = cleaner.handle_mixed_type()

            if 'split_params' in config:
                p = config['split_params']
                target = p['target']
                X = self.processed_df.drop(columns=[target])
                y = self.processed_df[target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=p.get('test_size', 0.2),
                    random_state=42,
                    stratify=y if p.get('stratify', False) else None
                )
                self.train_df = pd.concat([X_train, y_train], axis=1)
                self.test_df = pd.concat([X_test, y_test], axis=1)
                self.logs.append(f"Data split into Train ({len(X_train)}) and Test ({len(X_test)}).")
            else:
                self.train_df = self.processed_df.copy()
                self.test_df = None

            if 'impute_params' in config:
                p = config['impute_params']
                c_train = DataCleaner(self.train_df)
                self.train_df, fitted_imputer = c_train.impute_data(strategy=p['strategy'])
                if self.test_df is not None and fitted_imputer is not None:
                    num_cols = self.test_df.select_dtypes(include=['number']).columns.tolist()
                    self.test_df[num_cols] = fitted_imputer.transform(self.test_df[num_cols])
                self.logs.append(f"Imputed missing values using {p['strategy']}.")

            if 'outlier_params' in config:
                p = config['outlier_params']
                cols = p.get('columns', [])
                for col in cols:
                    h_train = OutlierHandler(self.train_df)
                    self.train_df = h_train.handle_outliers(col, method=p['method'], action=p['action'])
                    if self.test_df is not None:
                        h_test = OutlierHandler(self.test_df)
                        self.test_df = h_test.handle_outliers(col, method=p['method'], action=p['action'])
                self.logs.append(f"Handled outliers in {len(cols)} columns.")

            if 'encode_params' in config:
                p = config['encode_params']
                e_type = p['type']
                enc_train = DataEncoder(self.train_df)

                if e_type == 'target':
                    self.train_df, fitted_enc = enc_train.target_encode(
                        column=p.get('column'), target_column=p.get('target_column'), smoothing=p.get('smoothing', 10)
                    )
                    if self.test_df is not None and fitted_enc is not None:
                        smooth_values, global_mean = fitted_enc
                        col = p.get('column')
                        self.test_df[col] = self.test_df[col].map(smooth_values).fillna(global_mean)
                else:
                    func_name = f"{e_type}_encode"
                    self.train_df, fitted_enc = getattr(enc_train, func_name)(columns=p.get('columns'))
                    if self.test_df is not None:
                        enc_test = DataEncoder(self.test_df)
                        if e_type == 'frequency' and fitted_enc:
                            for col, freq_map in fitted_enc.items():
                                self.test_df[col] = self.test_df[col].map(freq_map)
                        elif e_type == 'onehot':
                            self.test_df = pd.get_dummies(self.test_df, columns=p.get('columns'), drop_first=True)
                            self.test_df = self.test_df.reindex(columns=self.train_df.columns, fill_value=0)
                        else:
                            self.test_df, _ = getattr(enc_test, func_name)(columns=p.get('columns'))
                self.logs.append(f"Applied {e_type} encoding.")

            if 'selection_params' in config:
                p = config['selection_params']

                if p.get('use_variance'):
                    selector = FeatureSelector(self.train_df)
                    self.train_df, dropped = selector.remove_low_variance(threshold=p.get('var_threshold', 0.01))
                    if self.test_df is not None:
                        self.test_df = self.test_df.drop(columns=dropped, errors='ignore')
                    self.logs.append(f"Variance Filter: Dropped {len(dropped)} features.")

                if p.get('use_correlation'):
                    selector = FeatureSelector(self.train_df)
                    self.train_df = selector.correlation_filter(threshold=p.get('corr_threshold', 0.9))
                    if self.test_df is not None:
                        self.test_df = self.test_df[self.train_df.columns]
                    self.logs.append("Correlation Filter applied.")

                method = p.get('method')
                if method:
                    selector = FeatureSelector(self.train_df)
                    if method == 'rfe':
                        self.train_df = selector.rfe_selection(target_col=p['target'], n_features=p.get('n_features', 5))
                    elif method == 'mutual_info':
                        self.train_df = selector.mutual_info_selection(target_col=p['target'], k=p.get('k', 5), task=p.get('task', 'classification'))
                    elif method == 'lasso':
                        self.train_df = selector.lasso_selection(target_col=p['target'])
                    elif method == 'chi2':
                        self.train_df = selector.chi2_selection(target_col=p['target'], k=p.get('k', 5))
                    if self.test_df is not None:
                        self.test_df = self.test_df[self.train_df.columns]
                    self.logs.append(f"Feature Selection: Applied {method}.")

            if 'balance_params' in config:
                p = config['balance_params']
                self.train_df = DataBalancer(self.train_df).balance_data(target_col=p['target'], method=p['method'])
                self.logs.append(f"Balanced Training set using {p['method']}.")

            if 'scale_params' in config:
                p = config['scale_params']
                s_train = DataScaler(self.train_df)
                self.train_df, fitted_scaler = s_train.scale_data(method=p['method'])
                if self.test_df is not None and fitted_scaler is not None:
                    num_cols = self.test_df.select_dtypes(include=['number']).columns.tolist()
                    self.test_df[num_cols] = fitted_scaler.transform(self.test_df[num_cols])
                self.logs.append(f"Applied {p['method']} scaling.")

            if 'transform_params' in config:
                p = config['transform_params']
                t_type = p['type']
                cols = p.get('columns', [])
                if t_type not in VALID_TRANSFORMS:
                    raise ValueError(f"Unknown transform type '{t_type}'. Valid options: {VALID_TRANSFORMS}")
                if t_type == 'log' and (self.train_df[cols] <= 0).any().any():
                    self.logs.append("Warning: Non-positive values detected in columns before log transform.")
                result = getattr(DataTransformer(self.train_df), f"apply_{t_type}_transform")(cols)
                if isinstance(result, tuple):
                    self.train_df, fitted_transformer = result
                else:
                    self.train_df, fitted_transformer = result, None
                if self.test_df is not None:
                    if t_type == 'power' and fitted_transformer is not None:
                        self.test_df[cols] = fitted_transformer.transform(self.test_df[cols])
                    else:
                        self.test_df = getattr(DataTransformer(self.test_df), f"apply_{t_type}_transform")(cols)
                self.logs.append(f"Applied {t_type} transformation to {cols}.")

            if self.test_df is not None:
                return (self.train_df, self.test_df), self.logs
            return self.train_df, self.logs

        except Exception as e:
            self.logs.append(f"Error in Pipeline: {str(e)}")
            return None, self.logs