import unicodedata
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re  # regular expressions
from const.java_tags import *
from const.java_keywords import *
import re


class DataProcesser:
    def remove_accented_chars(text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(text: str, remove_digits=True) -> str:
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def replace_special_characters(text: str, remove_digits=True) -> str:
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, ' ', text)
        return text

    def stemming(text: pd.Series):
        ps = PorterStemmer()
        stemmed = [(" ".join(list(map(ps.stem, comment.split())))) for comment in text]
        return pd.Series(stemmed)

    def stem(text: str):
        ps = PorterStemmer()
        stemmed = (" ".join(list(map(ps.stem, text.split()))))
        return stemmed

    def remove_stopwords(text: str) -> str:
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')

        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_java_tags(text: str) -> str:
        for tag in JAVA_TAGS:
            text = re.sub(rf"{tag.lower()}", '', text)
        return text

    def remove_java_keywords(text: str) -> str:
        for tag in JAVA_KEYWORDS:
            text = re.sub(rf"{tag.lower()}", '', text)
        return text

    def is_lower_case(ch):
        ch.lower() == ch

    def is_upper_case(ch):
        ch.upper() == ch

    def extract_camel_case(text: str) -> str:
        counter = len(text)
        i = 1
        while i < counter:
            prev = text[i - 1]
            cur = text[i]
            if prev.islower() and cur.isupper():
                text = text[0:i] + " " + cur + text[(i + 1):]
                i += 1
                counter += 1
            i += 1

        return text

    def extract_snake_case(text):
        text =  text.replace('_', ' ')
        return text

    def preprocess(text):
        # to lower case
        text = str(text)
        # text = DataProcesser.extract_camel_case(text)
        # text = DataProcesser.extract_snake_case(text)
        text = text.lower()
        #text = DataProcesser.remove_stopwords(text)
        text = DataProcesser.remove_java_tags(text)
        text = DataProcesser.remove_java_keywords(text)
        text = DataProcesser.remove_accented_chars(text)
        text = DataProcesser.replace_special_characters(text)
        text = DataProcesser.stem(text)
        return text

    def preprocess_code(text):
        # to lower case
        text = str(text)
        text = DataProcesser.extract_camel_case(text)
        text = DataProcesser.extract_snake_case(text)
        text = text.lower()
        text = DataProcesser.remove_java_tags(text)
        text = DataProcesser.remove_java_keywords(text)
        text = DataProcesser.remove_accented_chars(text)
        text = DataProcesser.replace_special_characters(text)
        #text = self.remove_stopwords(text)
        text = DataProcesser.stem(text)
        return text



