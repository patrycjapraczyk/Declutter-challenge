from helpers.data_loader import DataLoader
from helpers.data_preprocessing import DataProcesser
from helpers.feature_helper import FeatureHelper
from helpers.text_representation import TextRepresentation
from helpers.text_similarity import TextSimilarity

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, normalize, scale
from scipy import sparse


class LogisticRegressionHelper:
    def __init__(self):
        self.data = DataLoader.load_csv_file("./../data/train_set_0520.csv", ['type', 'comment', 'non-information'])
        code = DataLoader.load_csv_file("./../data/code_data.csv", ['code'])
        self.data['code'] = code['code']
        self.data['code'] = self.data['code'].apply(str)
        self.data['comment'] = self.data['comment'].apply(str)
        self.data['code'] = self.data['code'].apply(DataProcesser.preprocess_code)
        self.code = self.data['code']

        self.values = self.data['non-information'].values
        self.values = np.where(self.values == 'yes', 1, 0)

    def transform_type(self, text: str):
        if text == "Javadoc":
            return 0
        elif text == "Block":
            return 1
        else:
            return 2

    def preprocess_comment_data(self):
        self.data['comment'] = self.data['comment'].apply(str)
        self.java_tags_ratio = self.data.apply(lambda row: FeatureHelper.get_java_tags_ratio(row['comment']), axis=1).to_numpy()

        self.data['comment'] = self.data['comment']
        self.data['comment'] = self.data['comment'].apply(DataProcesser.preprocess)
        self.comments = self.data['comment']

    def extract_features(self):
        self.length_data = self.data['comment'].apply(lambda c: len(c.split())).to_numpy()
        self.stopwords_num = self.data['comment'].apply(FeatureHelper.get_stop_words_num).to_numpy()
        self.types = self.data['type'].apply(self.transform_type).to_numpy()
        self.code_comment_similarity = self.data.apply(lambda row:
                                                       TextSimilarity.get_similarity_score(
                                                           DataProcesser.preprocess(row['comment']),
                                                           DataProcesser.preprocess(row['code']), 'JACC'), axis=1).to_numpy()
        m = 0

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.values, test_size=0.25, random_state=1000)

    def vectorise_comment_data(self):
        self.comments = TextRepresentation.vectorize(self.comments, 'BOW')
        self.code_vectorised = TextRepresentation.vectorize(self.code, 'BOW')

    def combine_features(self):
        features = scale(self.length_data.reshape((self.length_data.shape[0], 1)))
        features = scale(np.hstack((features, self.stopwords_num.reshape(self.stopwords_num.shape[0], 1))))
        features = scale(np.hstack((features, self.code_comment_similarity.reshape(self.code_comment_similarity.shape[0], 1))))
        features = scale(np.hstack((features, self.java_tags_ratio.reshape(self.java_tags_ratio.shape[0], 1))))
        #features = scale(np.hstack((features, self.types.reshape(self.types.shape[0], 1))))

        features = sparse.hstack((features, self.comments))
        #features = scale(np.hstack((features, self.code_comment_similarity.reshape(self.code_comment_similarity.shape[0], 1))))
        #features = sparse.hstack((features, self.code_vectorised))
        self.features = features


    def fit_model(self):
        self.classifier = LogisticRegression()
        self.classifier.fit(self.x_train, self.y_train)

    def print_score(self):
        y_pred = self.classifier.predict(self.x_test)
        # Model Evaluation metrics
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        print('Accuracy Score : ' + str(accuracy_score(self.y_test, y_pred)))
        print('Precision Score : ' + str(precision_score(self.y_test, y_pred)))
        print('Recall Score : ' + str(recall_score(self.y_test, y_pred)))
        print('F1 Score : ' + str(f1_score(self.y_test, y_pred)))


def run():
    logistic_regression = LogisticRegressionHelper()
    logistic_regression.preprocess_comment_data()
    logistic_regression.vectorise_comment_data()
    logistic_regression.extract_features()
    logistic_regression.combine_features()
    logistic_regression.split_data()
    logistic_regression.fit_model()
    logistic_regression.print_score()

run()

