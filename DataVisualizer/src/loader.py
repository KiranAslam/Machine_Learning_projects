import pandas as pd
import os

class DataLoader:
    @staticmethod
    def load_data(file_path_or_buffer) -> pd.DataFrame:
        
        try:
            if isinstance(file_path_or_buffer, str):
                ext = os.path.splitext(file_path_or_buffer)[-1].lower()
            else:
                ext = os.path.splitext(file_path_or_buffer.name)[-1].lower()

            if ext == '.csv':
                return pd.read_csv(file_path_or_buffer)
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")