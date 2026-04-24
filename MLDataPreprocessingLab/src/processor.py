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
        self.train_df     = None
        self.test_df      = None
        self.logs         = []

    def Pipeline(self, df, config):
        self.logs = []
        if not config:
            self.logs.append("No processing steps selected. Returning raw data.")
            return df, self.logs

        self.processed_df = df.copy()

        try:
            # STEP 0 — Drop Columns
            if 'drop_columns' in config:
                cols_to_drop = [c for c in config['drop_columns'] if c in self.processed_df.columns]
                self.processed_df = self.processed_df.drop(columns=cols_to_drop)
                self.logs.append(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}.")

            # STEP 0b — Data Type Conversion
            if 'dtype_conversions' in config:
                for col, new_type in config['dtype_conversions'].items():
                    if col not in self.processed_df.columns:
                        continue
                    try:
                        if new_type == 'datetime':
                            self.processed_df[col] = pd.to_datetime(self.processed_df[col], errors='coerce')
                        elif new_type == 'numeric':
                            self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce')
                        elif new_type == 'category':
                            self.processed_df[col] = self.processed_df[col].astype('category')
                        elif new_type == 'string':
                            self.processed_df[col] = self.processed_df[col].astype(str)
                        self.logs.append(f"Converted {col} to {new_type}.")
                    except Exception as e:
                        self.logs.append(f"Warning: Could not convert {col} to {new_type}: {e}")

            # STEP 0c — Feature Engineering
            if 'feature_engineering' in config:
                for feat in config['feature_engineering']:
                    name    = feat['name']
                    columns = feat['columns']
                    op      = feat['op']
                    try:
                        # fill nulls with 0 before combining so missing values don't propagate
                        filled = [self.processed_df[c].fillna(0) for c in columns if c in self.processed_df.columns]
                        if not filled:
                            self.logs.append(f"Warning: No valid columns for feature {name}.")
                            continue
                        result_col = filled[0]
                        for s in filled[1:]:
                            if op == '+': result_col = result_col + s
                            elif op == '-': result_col = result_col - s
                            elif op == '*': result_col = result_col * s
                            elif op == '/': result_col = result_col / s.replace(0, np.nan)
                        self.processed_df[name] = result_col
                        expr = f" {op} ".join(columns)
                        self.logs.append(f"Created feature '{name}' = {expr}.")
                    except Exception as e:
                        self.logs.append(f"Warning: Feature engineering failed for {name}: {e}")

            # STEP 1 — Remove Duplicates
            if config.get('remove_duplicates'):
                dup_count = self.processed_df.duplicated().sum()
                self.processed_df = self.processed_df.drop_duplicates()
                self.logs.append(f"Removed {dup_count} duplicate rows.")

            # handle_mixed_type on full df before split
            cleaner = DataCleaner(self.processed_df)
            self.processed_df = cleaner.handle_mixed_type()

            # STEP 9 — Split (must happen before fitting anything)
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
                self.test_df  = pd.concat([X_test,  y_test],  axis=1)
                self.logs.append(f"Data split into Train ({len(X_train):,}) and Test ({len(X_test):,}).")
            else:
                self.train_df = self.processed_df.copy()
                self.test_df  = None

            # STEP 2 — Impute per column
            if 'impute_params' in config:
                # config is now {col: {strategy, n_neighbors}}
                for col, p in config['impute_params'].items():
                    if col not in self.train_df.columns:
                        continue
                    c_train = DataCleaner(self.train_df[[col]])
                    _, fitted_imputer = c_train.impute_data(strategy=p['strategy'], n_neighbors=p.get('n_neighbors', 5))
                    if fitted_imputer is None:
                        continue
                    self.train_df[[col]] = fitted_imputer.transform(self.train_df[[col]])
                    if self.test_df is not None and col in self.test_df.columns:
                        # apply train's fitted imputer to test — no refitting
                        self.test_df[[col]] = fitted_imputer.transform(self.test_df[[col]])
                    self.logs.append(f"Imputed '{col}' using {p['strategy']}.")

            # STEP 3 — Outliers
            if 'outlier_params' in config:
                p = config['outlier_params']
                for col in p.get('columns', []):
                    h = OutlierHandler(self.train_df)
                    self.train_df = h.handle_outliers(col, method=p['method'], action=p['action'])
                    if self.test_df is not None:
                        h2 = OutlierHandler(self.test_df)
                        self.test_df = h2.handle_outliers(col, method=p['method'], action=p['action'])
                self.logs.append(f"Handled outliers in {len(p.get('columns', []))} columns.")

            # STEP 4 — Encode per column
            if 'encode_params' in config:
                # config is now {col: {type, ...}}
                for col, p in config['encode_params'].items():
                    if col not in self.train_df.columns:
                        continue
                    e_type = p['type']
                    enc_train = DataEncoder(self.train_df)

                    if e_type == 'target':
                        self.train_df, fitted_enc = enc_train.target_encode(
                            column=col,
                            target_column=p.get('target_column'),
                            smoothing=p.get('smoothing', 10)
                        )
                        if self.test_df is not None and fitted_enc is not None:
                            smooth_values, global_mean = fitted_enc
                            self.test_df[col] = self.test_df[col].map(smooth_values).fillna(global_mean)

                    elif e_type == 'onehot':
                        self.train_df, fitted_enc = enc_train.one_hot_encode(columns=[col])
                        if self.test_df is not None and fitted_enc is not None:
                            # use train fitted sklearn encoder — handle_unknown ignore handles unseen categories
                            encoded = fitted_enc.transform(self.test_df[[col]])
                            feature_names = fitted_enc.get_feature_names_out([col])
                            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=self.test_df.index)
                            self.test_df = self.test_df.drop(columns=[col])
                            self.test_df = pd.concat([self.test_df, encoded_df], axis=1)
                            self.test_df = self.test_df.reindex(columns=self.train_df.columns, fill_value=0)

                    elif e_type == 'frequency':
                        self.train_df, freq_maps = enc_train.frequency_encode(columns=[col])
                        if self.test_df is not None and freq_maps:
                            self.test_df[col] = self.test_df[col].map(freq_maps[col])

                    else:
                        func_name = f"{e_type}_encode"
                        self.train_df, fitted_enc = getattr(enc_train, func_name)(columns=[col])
                        if self.test_df is not None:
                            enc_test = DataEncoder(self.test_df)
                            self.test_df, _ = getattr(enc_test, func_name)(columns=[col])

                    self.logs.append(f"Encoded '{col}' using {e_type}.")

            # STEP 5 — Feature Selection
            if 'selection_params' in config:
                p = config['selection_params']
                sel_target = p['target']

                # all selection methods require numeric features — warn and drop string cols before selection
                non_numeric_sel = [
                    c for c in self.train_df.columns
                    if c != sel_target and not pd.api.types.is_numeric_dtype(self.train_df[c])
                ]
                if non_numeric_sel:
                    self.logs.append(f"Warning: {len(non_numeric_sel)} non-numeric columns excluded from feature selection (encode them first): {non_numeric_sel}")
                    self.train_df = self.train_df.drop(columns=non_numeric_sel)
                    if self.test_df is not None:
                        self.test_df = self.test_df.drop(columns=[c for c in non_numeric_sel if c in self.test_df.columns])

                if p.get('use_variance'):
                    selector = FeatureSelector(self.train_df)
                    self.train_df, dropped = selector.remove_low_variance(threshold=p.get('var_threshold', 0.01))
                    if self.test_df is not None:
                        self.test_df = self.test_df.drop(columns=dropped, errors='ignore')
                    self.logs.append(f"Variance Filter: dropped {len(dropped)} features.")
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
                        self.train_df = selector.rfe_selection(target_col=sel_target, n_features=p.get('n_features', 5), task=p.get('task','classification'))
                    elif method == 'mutual_info':
                        self.train_df = selector.mutual_info_selection(target_col=sel_target, k=p.get('k', 5), task=p.get('task','classification'))
                    elif method == 'lasso':
                        self.train_df = selector.lasso_selection(target_col=sel_target)
                    elif method == 'chi2':
                        self.train_df = selector.chi2_selection(target_col=sel_target, k=p.get('k', 5))
                    if self.test_df is not None:
                        self.test_df = self.test_df[self.train_df.columns]
                    self.logs.append(f"Feature selection: {method} applied.")

            # STEP 6 — Balance (train only)
            if 'balance_params' in config:
                p = config['balance_params']
                target_col = p['target']

                # SMOTE/ADASYN cannot handle string columns — drop non-numeric non-target cols before balancing
                non_numeric = [
                    c for c in self.train_df.columns
                    if c != target_col and not pd.api.types.is_numeric_dtype(self.train_df[c])
                ]
                if non_numeric:
                    self.logs.append(f"Warning: Dropped {len(non_numeric)} non-numeric columns before balancing (encode them first): {non_numeric}")
                    self.train_df = self.train_df.drop(columns=non_numeric)
                    if self.test_df is not None:
                        self.test_df = self.test_df.drop(columns=[c for c in non_numeric if c in self.test_df.columns])

                self.train_df = DataBalancer(self.train_df).balance_data(target_col=target_col, method=p['method'])
                self.logs.append(f"Balanced training set using {p['method']}.")

            # STEP 7 — Scale
            if 'scale_params' in config:
                p = config['scale_params']
                # only scale numeric columns — store exactly which cols were fit so test uses same set
                train_num_cols = self.train_df.select_dtypes(include='number').columns.tolist()
                s_train = DataScaler(self.train_df)
                self.train_df, fitted_scaler = s_train.scale_data(columns=train_num_cols, method=p['method'])
                if self.test_df is not None and fitted_scaler is not None:
                    # use the exact same columns the scaler was fit on — not test's own numeric cols
                    test_scale_cols = [c for c in train_num_cols if c in self.test_df.columns]
                    self.test_df[test_scale_cols] = fitted_scaler.transform(self.test_df[test_scale_cols])
                self.logs.append(f"Applied {p['method']} scaling to {len(train_num_cols)} columns.")

            # STEP 8 — Transform per column
            if 'transform_params' in config:
                # config is now {col: {type, method?}}
                for col, p in config['transform_params'].items():
                    if col not in self.train_df.columns:
                        continue
                    t_type = p['type']
                    if t_type not in VALID_TRANSFORMS:
                        self.logs.append(f"Warning: Unknown transform type '{t_type}' for {col}, skipped.")
                        continue
                    if t_type == 'log' and (self.train_df[col] <= 0).any():
                        self.logs.append(f"Warning: Non-positive values in '{col}' before log transform.")

                    result = getattr(DataTransformer(self.train_df), f"apply_{t_type}_transform")([col])
                    if isinstance(result, tuple):
                        self.train_df, fitted_transformer = result
                    else:
                        self.train_df, fitted_transformer = result, None

                    if self.test_df is not None and col in self.test_df.columns:
                        if t_type == 'power' and fitted_transformer is not None:
                            self.test_df[[col]] = fitted_transformer.transform(self.test_df[[col]])
                        else:
                            self.test_df = getattr(DataTransformer(self.test_df), f"apply_{t_type}_transform")([col])
                            if isinstance(self.test_df, tuple):
                                self.test_df = self.test_df[0]

                    self.logs.append(f"Applied {t_type} transform to '{col}'.")

            if self.test_df is not None:
                return (self.train_df, self.test_df), self.logs
            return self.train_df, self.logs

        except Exception as e:
            self.logs.append(f"Error in Pipeline: {str(e)}")
            return None, self.logs