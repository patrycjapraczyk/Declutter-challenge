from helpers.data_loader import DataLoader
from helpers.data_preprocessing import DataProcesser
from helpers.feature_helper import FeatureHelper
from text_representation.text_representation_factory import TextRepresentationFactory
from models.model_factory import ModelFactory
from helpers.score_metrics import ScoreMetrics
from helpers.text_similarity import TextSimilarity
from models.ensemble import Ensemble
from helpers.class_imbalance_sampling import ImbalanceSampling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, normalize, scale
from scipy import sparse


class ModelExec:
    def __init__(self, include_comments=False, include_long_code=False):
        self.data = DataLoader.load_data(True)
        self.data = self.preprocess_comment_data(self.data)

    def preprocess_comment_data(self, data):
        data['comment'] = data['comment'].apply(str)
        # self.data['java_tags_ratio'] = self.data.apply(lambda row: FeatureHelper.get_java_tags_ratio(row['comment']),
        #                                        axis=1).to_numpy()
        data['comment'] = data['comment'].apply(DataProcesser.preprocess)
        data['code'] = data['code'].apply(DataProcesser.preprocess_code)
        return data

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.data, self.data['non-information'], test_size=0.25, random_state=1000)
        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def kfold_split(self, folds_split, w1, w2):
        i = 1
        kFold = KFold(n_splits=folds_split, shuffle=True, random_state=None)
        model_list = ModelFactory.get_models_list()
        model_list.append('ensemble')
        vals = [[name, [], [], [], [], [], []] for name in model_list]
        result = pd.DataFrame(columns=['name', 'accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'balanced_accuracy'], data=vals)

        for train_index, test_index in kFold.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            features_test = self.extract_features(test_data)
            features_train = self.extract_features(train_data)

            comment_vectorised = self.vectorise_comment_data(features_train[5], features_test[5])
            comments_train = comment_vectorised[0]
            comments_test = comment_vectorised[1]

            features_test = self.combine_features(features_test, include_comments=False)
            features_train = self.combine_features(features_train, include_comments=False)

            ensemble_result = self.ensemble_model(features_train, features_test,
                                                  comments_train, comments_test, train_data['non-information'],
                                                  test_data['non-information'], 0.5714285714285715, 0.4285714285714286)
            res = self.compare_models(features_train, features_test, train_data['non-information'],
                                test_data['non-information'])
            counter = 0
            for index, row in res.iterrows():
                result.iloc[counter]['accuracy'].append(row['accuracy'])
                result.iloc[counter]['precision'].append(row['precision'])
                result.iloc[counter]['recall'].append(row['recall'])
                result.iloc[counter]['f1'].append(row['f1'])
                result.iloc[counter]['matthews_corrcoef'].append(row['matthews_corrcoef'])
                result.iloc[counter]['balanced_accuracy'].append(row['balanced_accuracy'])
                counter += 1

            result.iloc[counter]['accuracy'].append(ensemble_result['accuracy'][0])
            result.iloc[counter]['precision'].append(ensemble_result['precision'][0])
            result.iloc[counter]['recall'].append(ensemble_result['recall'][0])
            result.iloc[counter]['f1'].append(ensemble_result['f1'][0])

        print('K fold')
        for index, row in result.iterrows():
            print(row['name'])
            acc = np.mean(row['accuracy'])
            print('Accuracy ' + str(acc))
            precision = np.mean(row['precision'])
            print('Precision ' + str(precision))
            recall = np.mean(row['recall'])
            print('Recall ' + str(recall))
            f1 = np.mean(row['f1'])
            print('F1 ' + str(f1))
            balanced_accuracy_score = np.mean(row['balanced_accuracy'])
            print('balanced_accuracy_score ' + str(balanced_accuracy_score))
            matthews_corrcoef = np.mean(row['matthews_corrcoef'])
            print('Matthews_corrcoef ' + str(matthews_corrcoef) + '\n')

        return result

    def extract_features(self, x):
        length_data = x['comment'].apply(lambda c: len(c.split())).to_numpy()
        stopwords_num = x['comment'].apply(FeatureHelper.get_stop_words_num).to_numpy()
        functional_types = x['functional_type'].to_numpy()
        #java_tags = x['java_tags_ratio'].to_numpy()
        x['comment'] = x['comment'].apply(DataProcesser.remove_stopwords)
        code_comment_similarity = x.apply(lambda row: TextSimilarity.get_similarity_score(
                                                          s1=DataProcesser.preprocess(row['comment']),
                                                          s2=DataProcesser.preprocess(row['code']),
                                                          type='COSINE_TFIDF'),
                                                      axis=1).to_numpy()

        comment = x['comment']
        features = [length_data, stopwords_num, code_comment_similarity, functional_types, comment]
        return features

    def vectorise_comment_data(self, comments_train, comment_test):
        text_representation = TextRepresentationFactory.get_text_representation('BOW', comments_train)
        x_train_comments = text_representation.vectorize(comments_train)
        x_test_comments = text_representation.vectorize(comment_test)
        return (x_train_comments, x_test_comments)


    def combine_features(self, feature_list: list, include_comments: bool) -> pd.DataFrame:
        features = scale(feature_list[0].reshape((feature_list[0].shape[0], 1)))
        features = scale(np.hstack((features, feature_list[1].reshape(feature_list[1].shape[0], 1))))
        features = scale(np.hstack((features, feature_list[2].reshape(feature_list[2].shape[0], 1))))
        features = scale(np.hstack((features, feature_list[3].reshape(feature_list[3].shape[0], 1))))
        #features = scale(np.hstack((features, feature_list[5].reshape(feature_list[5].shape[0], 1))))
        if include_comments:
            comments = feature_list[3]
            features = sparse.hstack((comments, features))
        return features

    def compare_models(self, features_train, features_test, y_train, y_test):
        x_train = features_train
        #x_train, y_train = ImbalanceSampling.get_sampled_data('RANDOM_UNDERSAMPLE', x_train, y_train)
        model_names = ModelFactory.get_models_list()
        score_df = pd.DataFrame(columns=['name', 'accuracy', 'precision', 'recall', 'f1'])
        for name in model_names:
            model = ModelFactory.get_model(name)
            model.fit_model(x_train, y_train)
            y_pred = model.predict(features_test)
            score = ScoreMetrics.get_scores(name, y_test, y_pred)
            print('-------')
            print(name)
            ScoreMetrics.print_scores(y_test, y_pred)
            score_df = score_df.append(score)
        return score_df

    def ensemble_model(self, features_train, features_test, comments_train, comments_test, y_train, y_test, w1, w2):
        score = Ensemble.get_ensemble_score('AVERAGING', comments_train, features_train,
                     comments_test, features_test, y_train, y_test, w1, w2)
        return score

    def execute_model(self, name):
        model = ModelFactory.get_model(name)
        split_data = self.split_data()
        features = self.extract_features(split_data['x_train'], split_data['x_test'])
        features_train = self.combine_features(features['features_train'], False)
        features_test = self.combine_features(features['features_test'], False)
        x_train, y_train = ImbalanceSampling.get_sampled_data('RANDOM_UNDERSAMPLE', features_train, split_data['y_train'])
        model.fit_model(x_train, y_train)
        y_pred = model.predict(features_test)
        ScoreMetrics.print_scores(split_data['y_test'], y_pred)
        return model

    def test_python_data(self):
        python_data = DataLoader.load_data_python()
        python_data = self.preprocess_comment_data(python_data)
        features_train = self.extract_features(self.data)
        features_test = self.extract_features(python_data)
        # comment_vectorised = self.vectorise_comment_data(features_train[4], features_test[4])
        # comments_train = comment_vectorised[0]
        # comments_test = comment_vectorised[1]
        features_test = self.combine_features(features_test, include_comments=False)
        features_train = self.combine_features(features_train, include_comments=False)
        y_train = self.data['non-information']
        y_test = python_data['non-information']
        self.compare_models(features_train, features_test, y_train, y_test)


exec = ModelExec(include_comments=False, include_long_code=True)
#exec.kfold_split(10, 0.3, 0.7)
exec.test_python_data()
