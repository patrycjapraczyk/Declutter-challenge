from nltk.corpus import stopwords
from helpers.data_preprocessing import DataProcesser
from helpers.textual_analysis import *


class FeatureHelper:
    def get_stop_words_num(x: str) -> int:
        stop_words = set(stopwords.words('english'))
        x = x.split()
        return len(set(x) & stop_words)

    def get_common_words(comment: str, code: str)-> int:
        code = DataProcesser.preprocess_code(code)
        comment = DataProcesser.preprocess_code(comment)
        return count_common_words(code, comment)

