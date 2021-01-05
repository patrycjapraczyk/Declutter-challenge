from text_representation.abstract_text_representation import AbstractTextRepresentation
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.preprocessing import normalize


class Word2Vec(AbstractTextRepresentation):
    W2V_NUM_FEATURES = 1000

    def build_model(self, tokenized_train):
        self.w2v_model = gensim.models.Word2Vec(tokenized_train, size=self.W2V_NUM_FEATURES,
                                           window=100, min_count=2, sample=1e-3, sg=1, iter=5, workers=10)

    def document_vectorizer(self, corpus, num_features):
        vocabulary = set(self.w2v_model.wv.index2word)

        def average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,), dtype="float64")
            nwords = 0.
            for word in words:
                if word in vocabulary:
                    nwords = nwords + 1.
                    feature_vector = np.add(feature_vector, model.wv[word])
            if nwords:
                feature_vector = np.divide(feature_vector, nwords)
            return feature_vector

        features = [average_word_vectors(tokenized_sentence, self.w2v_model, vocabulary,
                                         num_features) for tokenized_sentence in corpus]
        return np.array(features)

    def vectorize(self, corpus):
        corpus = [sent for sent in corpus]
        self.build_model(corpus)
        avg_wv_train_features = self.document_vectorizer(corpus=corpus, num_features=self.W2V_NUM_FEATURES)
        avg_wv_train_features = normalize(avg_wv_train_features, norm='l1', axis=0)
        return avg_wv_train_features

    def print(self):
        pass
