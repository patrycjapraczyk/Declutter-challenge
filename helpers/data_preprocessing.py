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
    def remove_accented_chars(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(self, text: str, remove_digits=True) -> str:
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def replace_special_characters(self, text: str, remove_digits=True) -> str:
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, ' ', text)
        return text

    def stemming(self, text: pd.Series):
        ps = PorterStemmer()
        stemmed = [(" ".join(list(map(ps.stem, comment.split())))) for comment in text]
        return pd.Series(stemmed)

    def stem(self, text: str):
        ps = PorterStemmer()
        stemmed = (" ".join(list(map(ps.stem, text.split()))))
        return stemmed

    def remove_stopwords(self, text: str) -> str:
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')

        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_java_tags(self, text: str) -> str:
        for tag in JAVA_TAGS:
            text = re.sub(rf"{tag.lower()}", '', text)
        return text

    def remove_java_keywords(self, text: str) -> str:
        for tag in JAVA_KEYWORDS:
            text = re.sub(rf"{tag.lower()}", '', text)
        return text

    def is_lower_case(self, ch):
        ch.lower() == ch

    def is_upper_case(self, ch):
        ch.upper() == ch

    def extract_camel_case(self, text: str) -> str:
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

    def extract_snake_case(self, text):
        text =  text.replace('_', ' ')
        return text

    def preprocess(self, text):
        # to lower case
        text = text.lower()
        text = self.remove_java_tags(text)
        #text = self.remove_accented_chars(text)
        text = self.replace_special_characters(text)
        text = self.stem(text)

        # text = text.apply(lambda x: x.lower())
        # text = text.apply(lambda x: self.remove_java_tags(x))
        # text = text.apply(lambda x: self.remove_accented_chars(x))
        # text = text.map(lambda x: self.remove_special_characters(x))
        # #stemming
        # text = self.stemming(text)
        # # comments_train = [remove_stopwords(comment) for comment in comments_train]
        return text



