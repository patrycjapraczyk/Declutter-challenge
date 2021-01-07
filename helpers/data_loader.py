import pandas as pd
import numpy as np


class DataLoader:
    TRAIN_DATA_PATH = "./../data/train_set_0520.csv"
    CODE_PATH = "./../data/code_data.csv"
    
    def load_data() -> pd.DataFrame:
        data = DataLoader.load_csv_file(DataLoader.TRAIN_DATA_PATH, ['type', 'comment', 'non-information'])
        code = DataLoader.load_csv_file(DataLoader.CODE_PATH, ['code'])
        data['code'] = code['code']
        data['code'] = data['code'].apply(str)
        data['comment'] = data['comment'].apply(str)
        data['non-information'] = data['non-information'].values
        data['non-information'] = np.where(data['non-information'] == 'yes', 1, 0)
        return data

    def load_csv_file(file_name: str, cols) -> pd.DataFrame:
        data = pd.read_csv(file_name, usecols=cols)
        return data
