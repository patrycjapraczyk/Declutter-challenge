from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


class Tfidf(AbstractTextRepresentation):
    def vectorize(self, corpus):
        self.tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2',
                             use_idf=True, smooth_idf=True)
        self.val = self.tv.fit_transform(corpus)
        self.val = normalize(self.val, norm='l1', axis=0)
        return self.val

    def print(self):
        tt_matrix = self.tv.toarray()
        vocab = self.tv.get_feature_names()
        pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)
