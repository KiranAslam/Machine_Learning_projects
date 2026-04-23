import pandas as pd
import os

class DataLoader:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv']

    def load_file(self, uploaded_file):
        if uploaded_file is None:
            return None

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in self.supported_formats:
            return f"Unsupported format: {file_extension}. Supported: {', '.join(self.supported_formats)}"

        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == '.tsv':
                df = pd.read_csv(uploaded_file, sep='\t')
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file)
            elif file_extension == '.parquet':
                df = pd.read_parquet(uploaded_file)

            df.columns = df.columns.str.strip()
            if df.empty:
                return "File loaded but contains no data."

            return df

        except Exception as e:
            return f"Error loading file: {str(e)}"

    def get_basic_info(self, df):
        if df is None or isinstance(df, str):
            return None

        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "size": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "numerical_cols": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_cols": df.select_dtypes(include=['datetime']).columns.tolist(),
            "missing_cols": df.columns[df.isnull().any()].tolist()
        }