from helpers.data_preprocessing import DataProcesser
from helpers.textual_analysis import *


class TextSimilarity:
    def get_similarity_score(s1: str, s2: str, type: str, params={}) -> int:
        measure_function = get_similarity_measure(type)
        return measure_function(s1, s2, params)

def get_similarity_measure(format):
    if format == 'JACC':
        return _jacc_score
    else:
        raise ValueError(format)

def _jacc_score(s1: str, s2: str, params={})-> int:
    comment_word_len = len(s1.split())
    code_word_len = len(s2.split())
    common_word_num = count_common_words(s1, s2)
    if common_word_num == 0:
        return 0
    return common_word_num / (comment_word_len + code_word_len + common_word_num)
