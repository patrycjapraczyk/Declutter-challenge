import pandas as pd
import numpy as np


class DataLoader:
    TRAIN_DATA_PATH = "./../data/train_set_0520.csv"
    CODE_PATH = "./../data/code_data.csv"
    CODE_LONG_PATH = "./../data/code_data_long2.csv"

    @staticmethod
    def load_longer_code():
        f = open("./../data/code.txt", "r")
        Lines = f.readlines()
        code_long = []
        for line in Lines:
            line = line.rstrip()
            # print(line)
            code_long.append(line)
        f.close()
        return code_long

    @staticmethod
    def load_data(load_code_longer=False) -> pd.DataFrame:
        data = DataLoader.load_csv_file(DataLoader.TRAIN_DATA_PATH, ['type', 'comment', 'non-information'])
        code = DataLoader.load_csv_file(DataLoader.CODE_PATH, ['code'])
        data['code'] = code['code']

        if load_code_longer:
            code_long = DataLoader.load_longer_code()
            for index, row in data.iterrows():
                curr_code_long = code_long[index]
                if curr_code_long != '':
                    row['code'] = curr_code_long

        data['code'] = data['code'].apply(str)
        data['comment'] = data['comment'].apply(str)
        data['non-information'] = data['non-information'].values
        data['non-information'] = np.where(data['non-information'] == 'yes', 1, 0)
        return data

    @staticmethod
    def load_csv_file(file_name: str, cols) -> pd.DataFrame:
        data = pd.read_csv(file_name, usecols=cols)
        return data
