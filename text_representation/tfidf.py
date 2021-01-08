from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


class Tfidf(AbstractTextRepresentation):
    def __init__(self, corpus):
        self.create_model(corpus)

    def create_model(self, corpus):
        corpus = [w for w in corpus]
        self.vectorizer = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                                 use_idf=True, smooth_idf=True)
        self.vectorizer.fit(corpus)

    def vectorize(self, corpus):
        self.val = self.vectorizer.transform(corpus)
        self.val = normalize(self.val, norm='l1', axis=0)
        return self.val

    def print(self):
        tt_matrix = self.tv.toarray()
        vocab = self.tv.get_feature_names()
        pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)
