from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import normalize


class BagOfWords(AbstractTextRepresentation):
    def vectorize(self, corpus):
        corpus = [w for w in corpus]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)
        self.val = self.vectorizer.transform(corpus)
        self.val = normalize(self.val, norm='l1', axis=0)
        return self.val

    def print(self, val):
        matrix = self.val.toarray()
        vocab = self.vectorizer.get_feature_names()
        df = pd.DataFrame(matrix, columns=vocab)
        return df
