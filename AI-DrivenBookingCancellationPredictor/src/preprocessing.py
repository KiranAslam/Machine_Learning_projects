import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from xgboost import XGBClassifier


def get_pipeline(scale_pos_weight=1.0):

    numerical_features = [
        'lead_time', 'adr',
        'total_of_special_requests', 'required_car_parking_spaces'
    ]
    categorical_features = [
        'hotel', 'market_segment', 'deposit_type', 'customer_type'
    ]
    high_card_features = ['country']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('target', ce.TargetEncoder(smoothing=10))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num',      numeric_transformer,      numerical_features),
        ('cat',      categorical_transformer,  categorical_features),
        ('high_card', high_card_transformer,   high_card_features)
    ])

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,   
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote',        SMOTE(random_state=42, k_neighbors=5)),
        ('classifier',   xgb)
    ])

    return full_pipeline