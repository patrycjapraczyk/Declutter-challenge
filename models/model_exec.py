from helpers.data_loader import DataLoader
from helpers.data_preprocessing import DataProcesser
from helpers.feature_helper import FeatureHelper
from text_representation.text_representation_factory import TextRepresentationFactory
from models.model_factory import ModelFactory
from helpers.score_metrics import ScoreMetrics
from helpers.text_similarity import TextSimilarity

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, normalize, scale
from scipy import sparse


class ModelExec:
    def __init__(self):
        self.data = DataLoader.load_data()
        self.preprocess_comment_data()
        self.split_data()
        self.extract_features()
        self.compare_models()

    def preprocess_comment_data(self):
        self.data['comment'] = self.data['comment'].apply(str)
        self.java_tags_ratio = self.data.apply(lambda row: FeatureHelper.get_java_tags_ratio(row['comment']),
                                               axis=1).to_numpy()

        self.data['comment'] = self.data['comment']
        self.data['comment'] = self.data['comment'].apply(DataProcesser.preprocess)

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.data['non-information'], test_size=0.25, random_state=1000)

    def extract_features(self):
        length_data_train = self.x_train['comment'].apply(lambda c: len(c.split())).to_numpy()
        stopwords_num_train = self.x_train['comment'].apply(FeatureHelper.get_stop_words_num).to_numpy()

        length_data_test = self.x_test['comment'].apply(lambda c: len(c.split())).to_numpy()
        stopwords_num_test = self.x_test['comment'].apply(FeatureHelper.get_stop_words_num).to_numpy()

        comment_vectorised = self.vectorise_comment_data()
        comments_train = comment_vectorised[0]
        comments_test = comment_vectorised[1]

        code_comment_similarity_train = self.x_train.apply(lambda row:
                                                           TextSimilarity.get_similarity_score(
                                                               s1=DataProcesser.preprocess(row['comment']),
                                                               s2=DataProcesser.preprocess(row['code']), type='COSINE_TFIDF'),
                                                           axis=1).to_numpy()

        code_comment_similarity_test = self.x_test.apply(lambda row:
                                                         TextSimilarity.get_similarity_score(
                                                             DataProcesser.preprocess(row['comment']),
                                                             DataProcesser.preprocess(row['code']), 'COSINE_TFIDF'),
                                                         axis=1).to_numpy()

        features_train = [length_data_train, stopwords_num_train, code_comment_similarity_train, comments_train]
        features_test = [length_data_test, stopwords_num_test, code_comment_similarity_test, comments_test]

        self.features_train = self.combine_features(features_train)
        self.features_test = self.combine_features(features_test)

    def vectorise_comment_data(self):
        text_representation = TextRepresentationFactory.get_text_representation('BOW', self.x_train['comment'])
        x_train_comments = text_representation.vectorize(self.x_train['comment'])
        x_test_comments = text_representation.vectorize(self.x_test['comment'])
        return (x_train_comments, x_test_comments)

    def combine_features(self, feature_list: list) -> pd.DataFrame:
        # TODO: stack comment
        features = scale(feature_list[0].reshape((feature_list[0].shape[0], 1)))
        features = scale(np.hstack((features, feature_list[1].reshape(feature_list[1].shape[0], 1))))
        features = scale(np.hstack((features, feature_list[2].reshape(feature_list[2].shape[0], 1))))
        #comments = feature_list[2]
        #features = scale(np.hstack((features, comments)))

        #features = feature_list[2]

        return features

    def compare_models(self):
        model_names = ModelFactory.get_models_list()
        score_df = pd.DataFrame(columns=['name', 'accuracy', 'precision', 'recall', 'f1'])
        for name in model_names:
            print('\n' + name)
            model = ModelFactory.get_model(name)
            model.fit_model(self.features_train, self.y_train)
            y_pred = model.predict(self.features_test)
            ScoreMetrics.print_scores(self.y_test, y_pred)
            score = ScoreMetrics.get_scores(name, self.y_test, y_pred)
            score_df.append(score)
        return score_df


ModelExec()
