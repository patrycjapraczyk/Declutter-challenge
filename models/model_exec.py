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
    def __init__(self, comment_vectoriser='BOW', imbalance_sampling=None):
        self.imbalance_sampling = imbalance_sampling
        self.data = DataLoader.load_data(load_code_longer=True)
        self.data = self.preprocess_data(self.data)
        self.comment_vectoriser = comment_vectoriser

    def preprocess_data(self, data):
        data['comment'] = data['comment'].apply(str)
        data['comment'] = data['comment'].apply(DataProcesser.preprocess)
        data['code'] = data['code'].apply(str)
        data['code'] = data['code'].apply(DataProcesser.preprocess_code)
        return data

    def split_data(self):
        """
        splits data into test and train set with the proportions of 75% and 25%
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.data, self.data['non-information'], test_size=0.25, random_state=1000)
        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def kfold_validate(self, splits, repeat):
        """
        Returns predictions from a repeated k-fold split
        Used for statistical tests
        """
        model_names = ModelFactory.get_models_list()
        result = pd.DataFrame(columns=["actual"] + model_names)
        for i in range(0, repeat):
            kFold = KFold(n_splits=splits, shuffle=True, random_state=None)
            for train_index, test_index in kFold.split(self.data):
                train_data = self.data.iloc[train_index]
                test_data = self.data.iloc[test_index]
                y_train = train_data['non-information']
                y_test = test_data['non-information']
                features_test = self.extract_features(test_data)
                features_train = self.extract_features(train_data)
                features_test = self.combine_features(features_test, comments_only=True)
                features_train = self.combine_features(features_train, comments_only=True)

                data = {'actual': y_test.tolist() }
                for model_name in model_names:
                    y_pred = self.execute_model_data(model_name, features_train, y_train, features_test)
                    data[model_name] = y_pred.tolist()
                df = pd.DataFrame(data=data)
                result = result.append(df)
        return result

    def kfold_split(self, folds_split, w1, w2, data):
        """
        Executes k-fold procedure on models of all types and prints final averaged results for all models.
        """
        i = 1
        kFold = KFold(n_splits=folds_split, shuffle=True, random_state=None)
        model_list = ModelFactory.get_models_list()
        model_list.append('ensemble')
        vals = [[name, [], [], [], [], [], [], []] for name in model_list]
        result = pd.DataFrame(
            columns=['name', 'accuracy', 'precision', 'recall', 'f1', 'matthews_corrcoef', 'balanced_accuracy',
                     'confusion_matrix'], data=vals)

        for train_index, test_index in kFold.split(self.data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]
            features_test = self.extract_features(test_data)
            features_train = self.extract_features(train_data)
            comment_vectorised = self.vectorise_comment_data(features_train[4], features_test[4])
            comments_train = comment_vectorised[0]
            comments_test = comment_vectorised[1]
            features_test = self.combine_features(features_test, comments_only=False)
            features_train = self.combine_features(features_train, comments_only=False)
            y_train = train_data['non-information']
            y_test = test_data['non-information']
            ensemble_result = self.ensemble_model(features_train, features_test,
                                                  comments_train, comments_test, y_train,
                                                  y_test, w1, w2)
            res = self.compare_models(comments_train, comments_test, train_data['non-information'],
                                       test_data['non-information'])
            counter = 0
            for index, row in res.iterrows():
                result.iloc[counter]['accuracy'].append(row['accuracy'])
                result.iloc[counter]['precision'].append(row['precision'])
                result.iloc[counter]['recall'].append(row['recall'])
                result.iloc[counter]['f1'].append(row['f1'])
                result.iloc[counter]['matthews_corrcoef'].append(row['matthews_corrcoef'])
                result.iloc[counter]['balanced_accuracy'].append(row['balanced_accuracy'])
                result.iloc[counter]['confusion_matrix'].append(row['confusion_matrix'])
                counter += 1
            acc = ensemble_result['accuracy'][0]
            result.iloc[counter]['accuracy'].append(acc)
            result.iloc[counter]['precision'].append(ensemble_result['precision'][0])
            result.iloc[counter]['recall'].append(ensemble_result['recall'][0])
            result.iloc[counter]['f1'].append(ensemble_result['f1'][0])
            result.iloc[counter]['matthews_corrcoef'].append(ensemble_result['matthews_corrcoef'][0])
            result.iloc[counter]['balanced_accuracy'].append(ensemble_result['balanced_accuracy'][0])
            result.iloc[counter]['confusion_matrix'].append(ensemble_result['confusion_matrix'][0])

        print('K fold')
        self.print_kfold_results(result)
        return result

    def print_kfold_results(self, result):
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
            print('Matthews_corrcoef ' + str(matthews_corrcoef))
            confusion_matrix = row['confusion_matrix']
            tn = 0
            fn = 0
            fp = 0
            tp = 0
            for curr in confusion_matrix:
                tn += curr[0][0]
                fn += curr[1][0]
                fp += curr[0][1]
                tp += curr[1][1]
            confusion_matrix = [[tn, fp], [fn, tp]]
            print('Confusion_matrix: \n ' + str(confusion_matrix) + '\n')

    def extract_features(self, x):
        """
        prepares all features and returns them as an array
        """
        length_data = x['comment'].apply(lambda c: len(c.split())).to_numpy()
        stopwords_num = x['comment'].apply(FeatureHelper.get_stop_words_num).to_numpy()
        functional_types = x['functional_type'].to_numpy()
        x['comment'] = x['comment'].apply(DataProcesser.remove_stopwords)
        code_comment_similarity = x.apply(lambda row: TextSimilarity.get_similarity_score(
            s1=DataProcesser.preprocess(row['comment']),
            s2=DataProcesser.preprocess(row['code']),
            type='JACC'),
                                          axis=1).to_numpy()
        code_comment_similarity_cosine = x.apply(lambda row: TextSimilarity.get_similarity_score(
            s1=DataProcesser.preprocess(row['comment']),
            s2=DataProcesser.preprocess(row['code']),
            type='COSINE_TFIDF'),
                                          axis=1).to_numpy()

        comment = x['comment'].to_numpy()
        features = [length_data, stopwords_num, code_comment_similarity_cosine, functional_types, comment]
        return features

    def vectorise_comment_data(self, comments_train, comment_test):
        text_representation = TextRepresentationFactory.get_text_representation(self.comment_vectoriser, comments_train)
        x_train_comments = text_representation.vectorize(comments_train)
        x_test_comments = text_representation.vectorize(comment_test)
        if self.comment_vectoriser != 'W2V':
            x_train_comments = x_train_comments.toarray()
            x_test_comments = x_test_comments.toarray()
        return (x_train_comments, x_test_comments)

    def combine_features(self, feature_list: list, comments_only: bool) -> pd.DataFrame:
        """
        Combines features into a single matrix, features are stacked together horizontally
        """
        features = scale(feature_list[0].reshape((feature_list[0].shape[0], 1)))
        features = scale(np.hstack((features, feature_list[1].reshape(feature_list[1].shape[0], 1))))
        features = scale(np.hstack((features, feature_list[2].reshape(feature_list[2].shape[0], 1))))
        features = scale(np.hstack((features, feature_list[3].reshape(feature_list[3].shape[0], 1))))
        if comments_only:
            comments = feature_list[4]
            features = comments
        return features

    def compare_models(self, features_train, features_test, y_train, y_test):
        """
        Executes models of all types implemented in this project and prints their results
        """
        x_train = features_train
        model_names = ModelFactory.get_models_list()
        score_df = pd.DataFrame(columns=['name', 'accuracy', 'precision', 'recall', 'f1'])
        if self.imbalance_sampling:
            x_train, y_train = ImbalanceSampling.get_sampled_data(self.imbalance_sampling, x_train, y_train)

        for name in model_names:
            model = ModelFactory.get_model(name, optimised=False)
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
                                            comments_test, features_test, y_train, y_test, w1, w2, self.imbalance_sampling)
        return score

    def execute_model_data(self, name, x_train, y_train, x_test):
        model = ModelFactory.get_model(name)
        model.fit_model(x_train, y_train)
        y_pred = model.predict(x_test)
        return y_pred

    def execute_model(self, name, imbalance_sampling=None):
        model = ModelFactory.get_model(name)
        split_data = self.split_data()
        features_train = self.extract_features(split_data['x_train'])
        features_test = self.extract_features(split_data['x_test'])
        features_train = self.combine_features(features_train, False)
        features_test = self.combine_features(features_test, False)
        x_train = features_train
        y_train = split_data['y_train']
        if imbalance_sampling:
            x_train, y_train = ImbalanceSampling.get_sampled_data(imbalance_sampling, x_train, y_train)
        model.fit_model(x_train, y_train)
        y_pred = model.predict(features_test)
        ScoreMetrics.print_scores(split_data['y_test'], y_pred)
        return model

    def test_python_data(self):
        python_data = DataLoader.load_data_python()
        python_data = self.preprocess_data(python_data)
        features_train = self.extract_features(self.data)
        features_test = self.extract_features(python_data)
        comment_vectorised = self.vectorise_comment_data(features_train[4], features_test[4])
        comments_train = comment_vectorised[0]
        comments_test = comment_vectorised[1]
        features_test = self.combine_features(features_test, comments_only=False)
        features_train = self.combine_features(features_train, comments_only=False)
        y_train = self.data['non-information']
        y_test = python_data['non-information']
        print("Testing modeels against Python data")
        self.compare_models(comments_train, comments_test, y_train, y_test)
        ensemble_result = self.ensemble_model(features_train, features_test,
                                              comments_train, comments_test, y_train,
                                              y_test, 0.5454545454545454, 0.45454545454545453)
        print('ENSEMBLE')
        print('acc: ', ensemble_result['accuracy'][0])
        print('precision: ', ensemble_result['precision'][0])
        print('recall: ', ensemble_result['recall'][0])
        print('f1: ', ensemble_result['f1'][0])
        print('matthews_corrcoef: ', ensemble_result['matthews_corrcoef'][0])
        print('balanced_accuracy: ', ensemble_result['balanced_accuracy'][0])
        print('confusion_matrix: ', ensemble_result['confusion_matrix'][0])
        print(ensemble_result)


