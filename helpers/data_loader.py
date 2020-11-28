import pandas as pd


class DataLoader:
    def load_csv_file(file_name: str, cols) -> pd.DataFrame:
        data = pd.read_csv(file_name, usecols=cols)
        return data
