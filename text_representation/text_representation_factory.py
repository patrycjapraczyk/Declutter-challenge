from text_representation.abstract_text_representation import AbstractTextRepresentation
from text_representation.bag_of_words import BagOfWords
from text_representation.bag_of_ngrams import BagOfNgrams
from text_representation.word2vec import Word2Vec
from text_representation.tfidf import Tfidf


class TextRepresentationFactory:
    def get_text_representation(format, corpus) -> AbstractTextRepresentation:
        if format == 'BOW':
            return BagOfWords(corpus)
        elif format == 'B-NGRAM':
            return BagOfNgrams(corpus)
        elif format == 'TFIDF':
            return Tfidf(corpus)
        elif format == 'W2V':
            return Word2Vec(corpus)
        else:
            raise ValueError(format)