from text_representation.abstract_text_representation import AbstractTextRepresentation
from text_representation.bag_of_words import BagOfWords
from text_representation.bag_of_ngrams import BagOfNgrams
from text_representation.word2vec import Word2Vec
from text_representation.tfidf import Tfidf


class TextRepresentationFactory:
    def get_text_representation(format) -> AbstractTextRepresentation:
        if format == 'BOW':
            return BagOfWords()
        elif format == 'B-NGRAM':
            return BagOfNgrams()
        elif format == 'TFIDF':
            return Tfidf()
        elif format == 'W2V':
            return Word2Vec()
        else:
            raise ValueError(format)