import pandas as pd
import numpy as np


class DataLoader:
    TRAIN_DATA_PATH = "./../data/train_set_0520.csv"
    CODE_PATH = "./../data/code_data.csv"
    CODE_LONG_PATH = "./../data/code_javadoc.txt"
    FUNCTIONAL_TYPES = "./../data/functional_types.csv"
    PYTHON_DATA = "./../data/python_data.csv"


    @staticmethod
    def load_longer_code():
        f = open(DataLoader.CODE_LONG_PATH, "r")
        Lines = f.readlines()
        code_long = []
        for line in Lines:
            line = line.rstrip()
            # print(line)
            code_long.append(line)
        f.close()
        return code_long

    @staticmethod
    def apply_functional_types(data):
        functional_types = ['method_declaration', 'class_declaration', 'assignment', 'method_call', 'return',
                            'requires', 'enum',
                            'loop', 'conditional', 'catch', 'var_declaration', 'package_import', 'loop_exit', 'empty']
        data['functional_type'] = data['functional_type'].apply(functional_types.index)
        i = 0

    @staticmethod
    def load_data(load_code_longer=False) -> pd.DataFrame:
        data = DataLoader.load_csv_file(DataLoader.TRAIN_DATA_PATH, ['type', 'comment', 'non-information'])
        code = DataLoader.load_csv_file(DataLoader.CODE_PATH, ['code'])
        functional_types = DataLoader.load_csv_file(DataLoader.FUNCTIONAL_TYPES, ['functional_type'])
        data['code'] = code['code']
        data['functional_type'] = functional_types['functional_type']
        DataLoader.apply_functional_types(data)

        if load_code_longer:
            code_long = DataLoader.load_longer_code()
            for index, row in data.iterrows():
                curr_code_long = code_long[index]
                if curr_code_long != '':
                    row['code'] = curr_code_long

        data['code'] = data['code'].apply(str)
        data['comment'] = data['comment'].apply(str)
        data['non-information'] = data['non-information'].values
        data['non-information'] = np.where(data['non-information'] == 'yes', 0, 1)
        return data

    @staticmethod
    def load_data_python() -> pd.DataFrame:
        data = DataLoader.load_csv_file(DataLoader.PYTHON_DATA, ['comment', 'non-information', 'code', 'functional_type'])
        DataLoader.apply_functional_types(data)
        data['code'] = data['code'].apply(str)
        data['comment'] = data['comment'].apply(str)
        data['non-information'] = data['non-information'].values
        data['non-information'] = np.where(data['non-information'] == 'yes', 0, 1)
        return data

    @staticmethod
    def load_csv_file(file_name: str, cols) -> pd.DataFrame:
        data = pd.read_csv(file_name, usecols=cols)
        return data
