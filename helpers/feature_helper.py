from nltk.corpus import stopwords
import re

from helpers.data_preprocessing import DataProcesser
from helpers.textual_analysis import *
from const.java_tags import *


class FeatureHelper:
    def get_stop_words_num(x: str) -> int:
        stop_words = set(stopwords.words('english'))
        x = x.split()
        length = len(x)
        stop_words_num = len(set(x) & stop_words)
        if stop_words_num == 0:
            return 0
        return stop_words_num / length

    def get_java_tags_num(text: str) -> int:
        tags = []
        text = str(text)
        text = text.lower()
        for tag in JAVA_TAGS:
            tags += re.findall(tag.lower(), text)
        return len(tags)

    def get_java_tags_ratio(text: str) -> int:
        tag_num = FeatureHelper.get_java_tags_num(text)
        for tag in JAVA_TAGS:
            text = re.sub(rf"{tag.lower()}", '', text)
        total_len = len(text.split())

        return tag_num / total_len + tag_num


