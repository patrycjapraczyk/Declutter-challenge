import unicodedata
from nltk.stem import PorterStemmer
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re  # regular expressions


class DataProcesser:
    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(self, text, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def stemming(self, text):
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

    def preprocess(self, text):
        text = text.map(self.remove_accented_chars)
        text = text.map(self.remove_special_characters)
        # to lower case
        text = text.map(lambda text: text.lower())
        # stemming
        text = self.stemming(text)
        # comments_train = [remove_stopwords(comment) for comment in comments_train]
        text = pd.Series(text)
        return text

    def preprocess_stopwords(self, text):
        return text.map(self.remove_stopwords)


