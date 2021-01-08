from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import normalize


class BagOfWords(AbstractTextRepresentation):
    def __init__(self, corpus):
        self.create_model(corpus)

    def create_model(self, corpus):
        corpus = [w for w in corpus]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

    def vectorize(self, val):
        self.val = self.vectorizer.transform(val)
        #self.val = normalize(self.val, norm='l1', axis=0)
        return self.val

    def print(self, val):
        matrix = self.val.toarray()
        vocab = self.vectorizer.get_feature_names()
        df = pd.DataFrame(matrix, columns=vocab)
        return df
