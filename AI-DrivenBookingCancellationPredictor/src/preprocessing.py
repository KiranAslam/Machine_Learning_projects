import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier



def get_pipeline():
    numerical_features =['lead_time', 'adr', 'total_of_special_requests', 'required_car_parking_spaces']
    categorical_features = ['hotel', 'market_segment', 'deposit_type', 'customer_type']
    high_card_features =['country']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('target', ce.TargetEncoder(smoothing=10))
    ])

    prepropcessors = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('high_card', high_card_transformer, high_card_features)
    ])

    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', prepropcessors),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100,random_state=42))
    ])

    return full_pipeline