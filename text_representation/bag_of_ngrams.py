from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import normalize


class BagOfNgrams(AbstractTextRepresentation):
    def vectorize(self, corpus):
        self.vectorizer = CountVectorizer(ngram_range=(2, 2))
        self.val = self.vectorizer.fit_transform(corpus)
        self.val = normalize(self.val, norm='l1', axis=0)
        return self.val

    def print(self):
        matrix = self.val.toarray()
        vocab = self.vectorizer.get_feature_names()
        pd.DataFrame(matrix, columns=vocab)
