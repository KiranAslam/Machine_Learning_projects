import pandas as pd 


class InsightRecommender:

    @staticmethod
    def generate_suggestions(health_summary: dict, missing_stats: pd.DataFrame, imbalance_info: dict) -> list:
        suggestions = []

        if health_summary['duplicate_count'] > 0:
            suggestions.append(f"Remove {health_summary['duplicate_count']} duplicate rows to avoid model bias.")

        for _, row in missing_stats.iterrows():
            if row['Percentage'] > 40:
                suggestions.append(f"Feature '{row['Column']}' has >40% missing values. Consider dropping it.")
            else:
                suggestions.append(f"Feature '{row['Column']}' has minor missingness. Use Median or KNN Imputation.")

        if imbalance_info.get('imbalance_detected'):
            suggestions.append("Significant class imbalance detected. Use SMOTE or class-weight adjustment during training.")

        return suggestions