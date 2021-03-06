from helpers.textual_analysis import *
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from text_representation.text_representation_factory import TextRepresentationFactory


class TextSimilarity:
    @staticmethod
    def get_similarity_score(s1: str, s2: str, type: str) -> int:
        """
        calculates the similarity score between s1 and s2
        type can be ‘JACC’, ‘COSINE’ or ‘COSINE_TFIDF’
        """
        measure_function = TextSimilarity.get_similarity_measure(type)
        return measure_function(s1, s2)

    @staticmethod
    def get_similarity_measure(format):
        if format == 'JACC':
            return TextSimilarity._jacc_score
        if format == 'COSINE':
            return TextSimilarity._cosine_similarity
        if format == 'COSINE_TFIDF':
            return TextSimilarity._cosine_similarity_tfidf
        else:
            raise ValueError(format)

    @staticmethod
    def _jacc_score(s1: str, s2: str) -> object:
        comment_words = s1.split()
        code_words = s2.split()
        intersection = len(list(set(comment_words).intersection(code_words)))
        union = (len(comment_words) + len(code_words)) - intersection
        return float(intersection) / union

    @staticmethod
    def _cosine_similarity(s1: str, s2: str):
        def text_to_vector(text):
            WORD = re.compile(r"\w+")
            words = WORD.findall(text)
            return Counter(words)
        vec1 = text_to_vector(s1)
        vec2 = text_to_vector(s2)

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        return cosine_similarity(df1, df2)

    @staticmethod
    def _cosine_similarity_tfidf(s1: str, s2: str):
        vectorizer = TextRepresentationFactory.get_text_representation('TFIDF', [s1, s2])
        tfidf = vectorizer.vectorize([s1, s2])
        return ((tfidf * tfidf.T).A)[0, 1]