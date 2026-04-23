import pandas as pd
import numpy as np

def get_detailed_audit(df):
    summary = {
        "Dimensions": f"{df.shape[0]} Rows x {df.shape[1]} Columns",
        "Total Cells": df.size,
        "Missing Cells": df.isnull().sum().sum(),
        "Missing %": f"{(df.isnull().sum().sum() / df.size * 100):.2f}%",
        "Duplicate Rows": df.duplicated().sum(),
        "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }

    health_checks = []
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        uniques = df[col].nunique()
        is_num = pd.api.types.is_numeric_dtype(df[col])
        null_perc = (nulls / len(df)) * 100 if len(df) > 0 else 0
        skew = "N/A"
        if is_num and len(df[col].dropna()) > 2:
            skew = round(df[col].skew(), 2)
        impute_sugg = "None"
        if null_perc > 50:
            impute_sugg = "Drop Column"
        elif null_perc > 0:
            impute_sugg = "KNN" if len(df) < 50000 else "Median/Mode"

        issues = []
        if nulls > 0:
            issues.append(f"{nulls} Missing")
        if uniques == 1:
            issues.append("Constant Column")
        if uniques > 100 and dtype == 'object':
            issues.append("High Cardinality")
        if is_num and skew != "N/A" and abs(skew) > 1:
            issues.append(f"Skewed ({skew})")

        outliers = 0
        if is_num:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                issues.append(f"{outliers} Outliers")

        health_checks.append({
            "Column": col,
            "Type": str(dtype),
            "Skewness": skew,
            "Unique Values": uniques,
            "Missing %": f"{null_perc:.1f}%",
            "Outliers": outliers,
            "Impute Suggestion": impute_sugg,
            "Status": ", ".join(issues) if issues else "Healthy"
        })

    return summary, pd.DataFrame(health_checks)


def get_smart_recommendations(df):
    recommendations = []

    for col in df.columns:
        null_perc = (df[col].isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
        if null_perc > 50:
            recommendations.append(
                f"**{col}**: High missing values ({null_perc:.1f}%). Suggestion: Drop this column."
            )
        elif null_perc > 0:
            suggestion = "KNN Imputer" if len(df) < 50000 else "Median/Mode (dataset too large for KNN)"
            recommendations.append(
                f"**{col}**: Has missing values ({null_perc:.1f}%). Suggestion: {suggestion}."
            )
        if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].dropna()) > 2:
            skew = df[col].skew()
            if abs(skew) > 1:
                transform = "Log or Box-Cox" if df[col].min() > 0 else "Yeo-Johnson Power Transform"
                recommendations.append(
                    f"**{col}**: Highly skewed ({skew:.2f}). Suggestion: Apply {transform}."
                )
        is_categorical = isinstance(df[col].dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(df[col]) or is_categorical:
            uniques = df[col].nunique()
            if uniques == 1:
                recommendations.append(
                    f"**{col}**: Constant column (only 1 unique value). Suggestion: Drop this column."
                )
            elif uniques == df.shape[0]:
                recommendations.append(
                    f"**{col}**: All values are unique. Likely an ID column. Suggestion: Drop before training."
                )
            elif uniques > 20:
                recommendations.append(
                    f"**{col}**: High cardinality ({uniques} labels). Suggestion: Use Target Encoding instead of One-Hot."
                )

    if not recommendations:
        recommendations.append("No major issues detected. Dataset looks ready for preprocessing.")

    return recommendations