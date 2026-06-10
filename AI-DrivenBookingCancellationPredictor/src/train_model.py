import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

from preprocessing import get_pipeline

def build_preprocessor_only():
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
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    high_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('target',  ce.TargetEncoder(smoothing=10))
    ])

    return ColumnTransformer(transformers=[
        ('num',       numeric_transformer,     numerical_features),
        ('cat',       categorical_transformer, categorical_features),
        ('high_card', high_card_transformer,   high_card_features)
    ])


def train_model(data_path='../Data/hotel_bookings.csv',
                model_output='../models/booking_cancellation_predictor.pkl',
                report_output='../models/training_report.json'):

    os.makedirs(os.path.dirname(model_output),  exist_ok=True)
    os.makedirs(os.path.dirname(report_output), exist_ok=True)

    print("[1/6] Loading data...")
    df = pd.read_csv(data_path)
    df.drop(
        columns=['reservation_status', 'reservation_status_date',
                 'company', 'agent'],
        inplace=True, errors='ignore'
    )
    df['total_guests'] = (
        df['adults'] + df['children'].fillna(0) + df['babies']
    )
    df['total_stays'] = (
        df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    )

    feature_cols = [
        'lead_time', 'adr', 'total_of_special_requests',
        'required_car_parking_spaces',
        'hotel', 'market_segment', 'deposit_type', 'customer_type', 'country'
    ]
    X = df[feature_cols].copy()
    Y = df['is_canceled']

    print("[2/6] Splitting data (80/20 stratified)...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    neg = (Y_train == 0).sum()
    pos = (Y_train == 1).sum()
    scale_pos_weight = round(neg / pos, 4)

    print(f"       Train size : {len(X_train):,}  |  "
          f"Cancellation rate: {Y_train.mean()*100:.1f}%")
    print(f"       scale_pos_weight (neg/pos): {scale_pos_weight}")

    print("[3/6] Training XGBoost pipeline (SMOTE + XGBClassifier)...")
    model_pipeline = get_pipeline(scale_pos_weight=scale_pos_weight)
    model_pipeline.fit(X_train, Y_train)

    print("[4/6] Evaluating on hold-out test set...")
    Y_pred      = model_pipeline.predict(X_test)
    Y_pred_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_prob)
    logloss = log_loss(Y_test, Y_pred_prob)
    mcc = matthews_corrcoef(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred).tolist()
    cls_rep = classification_report(Y_test, Y_pred, output_dict=True)

    print("[5/6] Extracting per-round training curves...")
    from imblearn.over_sampling import SMOTE as _SMOTE

    prep_only = build_preprocessor_only()
    X_train_t = prep_only.fit_transform(X_train, Y_train)
    X_test_t  = prep_only.transform(X_test)

    sm = _SMOTE(random_state=42, k_neighbors=5)
    X_sm, Y_sm = sm.fit_resample(X_train_t, Y_train)

    from xgboost import XGBClassifier
    eval_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric=['logloss', 'error', 'auc'],
        use_label_encoder=False, random_state=42,
        n_jobs=-1, verbosity=0
    )
    eval_model.fit(
        X_sm, Y_sm,
        eval_set=[(X_sm, Y_sm), (X_test_t, Y_test)],
        verbose=False
    )
    evals = eval_model.evals_result()
    train_logloss = evals['validation_0']['logloss']
    val_logloss   = evals['validation_1']['logloss']
    train_error   = evals['validation_0']['error']
    val_error     = evals['validation_1']['error']
    train_auc     = evals['validation_0']['auc']
    val_auc       = evals['validation_1']['auc']

    xgb_clf = model_pipeline.named_steps['classifier']
    fi_scores = xgb_clf.feature_importances_

    try:
        ct = model_pipeline.named_steps['preprocessor']
        feat_names = ct.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(len(fi_scores))]

    fi_pairs = sorted(
        zip(feat_names, fi_scores.tolist()),
        key=lambda x: x[1], reverse=True
    )[:15]
    feature_importance = [{"feature": f, "importance": round(v, 6)}
                          for f, v in fi_pairs]

    print("[5c] 5-fold cross-validation (this may take ~30s)...")
    cv_pipeline = get_pipeline(scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(cv_pipeline, X, Y, cv=cv,
                            scoring='f1', n_jobs=-1)
    cv_roc = cross_val_score(cv_pipeline, X, Y, cv=cv,
                             scoring='roc_auc', n_jobs=-1)


    print("[6/6] Saving model and report...")
    joblib.dump(model_pipeline, model_output)

    report = {
        "model": "XGBClassifier",
        "xgb_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight
        },
        "data": {
            "total_samples":      int(len(df)),
            "train_samples":      int(len(X_train)),
            "test_samples":       int(len(X_test)),
            "cancellation_rate":  round(float(Y.mean()) * 100, 2),
            "smote_applied":      True,
            "smote_train_samples_after": int(len(X_sm))
        },
        "holdout_metrics": {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
            "roc_auc":   round(roc_auc, 4),
            "log_loss":  round(logloss, 4),
            "mcc":       round(mcc, 4)
        },
        "confusion_matrix": {
            "TN": cm[0][0], "FP": cm[0][1],
            "FN": cm[1][0], "TP": cm[1][1]
        },
        "classification_report": cls_rep,
        "cross_validation": {
            "folds": 5,
            "f1_scores":    [round(x, 4) for x in cv_f1.tolist()],
            "f1_mean":      round(float(cv_f1.mean()), 4),
            "f1_std":       round(float(cv_f1.std()), 4),
            "roc_auc_scores": [round(x, 4) for x in cv_roc.tolist()],
            "roc_auc_mean": round(float(cv_roc.mean()), 4),
            "roc_auc_std":  round(float(cv_roc.std()), 4)
        },
        "training_curves": {
            "epochs":          list(range(1, 301)),
            "train_logloss":   [round(x, 6) for x in train_logloss],
            "val_logloss":     [round(x, 6) for x in val_logloss],
            "train_error":     [round(x, 6) for x in train_error],
            "val_error":       [round(x, 6) for x in val_error],
            "train_auc":       [round(x, 6) for x in train_auc],
            "val_auc":         [round(x, 6) for x in val_auc]
        },
        "feature_importance": feature_importance
    }

    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*55)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("="*55)
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1 Score   : {f1*100:.2f}%")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"  Log Loss   : {logloss:.4f}")
    print(f"  MCC        : {mcc:.4f}")
    print(f"  CV F1      : {cv_f1.mean()*100:.2f}% ± {cv_f1.std()*100:.2f}%")
    print(f"  CV ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    print("="*55)
    print(f"  Model  → {model_output}")
    print(f"  Report → {report_output}")
    print("="*55)

    return report


if __name__ == "__main__":
    train_model()