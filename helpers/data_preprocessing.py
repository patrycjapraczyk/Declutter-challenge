import unicodedata
from nltk.stem import PorterStemmer
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re  # regular expressions


class DataProcesser:
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(text, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def stemming(text):
        ps = PorterStemmer()
        return [(" ".join(list(map(ps.stem, comment.split())))) for comment in text]

    def remove_stopwords(text):
        tokenizer = ToktokTokenizer()
        stopword_list = nltk.corpus.stopwords.words('english')

        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
