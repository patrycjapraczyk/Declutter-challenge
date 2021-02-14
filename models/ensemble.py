from models.model_factory import ModelFactory
import numpy as np
from statistics import mode
from helpers.score_metrics import ScoreMetrics
from helpers.class_imbalance_sampling import ImbalanceSampling
from sklearn.preprocessing import LabelEncoder, normalize, scale



class Ensemble:
    @staticmethod
    def get_ensemble_score_all_features(mode, features_train, features_test, y_train, y_test):
        if mode == 'AVERAGING':
            return Ensemble.ensemble_averaging_all_features(features_train, features_test, y_train, y_test)

    @staticmethod
    def get_ensemble_score(mode, features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test):
        if mode == 'MAX_VOTE':
            return Ensemble.ensemble_max_vote(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test)
        elif mode == 'AVERAGING':
            return Ensemble.ensemble_averaging(features_train_text, features_train_notext,
                                       features_test_text, features_test_notext, y_train, y_test)

    @staticmethod
    def ensemble_max_vote(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test):
        model_notext = ModelFactory.get_model('LogisticRegression')
        model_text = ModelFactory.get_model('RandomForest')

        model_text.fit_model(features_train_text, y_train)
        model_notext.fit_model(features_train_notext, y_train)

        pred1 = model_text.predict(features_test_text)
        pred2 = model_notext.predict(features_test_notext)

        final_pred = np.array([])
        for i in range(0, len(y_test)):
            val = mode([pred1[i], pred2[i]])
            final_pred = np.append(final_pred, val)
        return ScoreMetrics.get_scores(y_test, final_pred)

    @staticmethod
    def ensemble_averaging(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test):
        model_notext = ModelFactory.get_model('SVM')
        model_text = ModelFactory.get_model('RandomForest')

        model_notext.fit_model(features_train_notext, y_train)
        model_text.fit_model(features_train_text, y_train)

        model_notext_balanced = ModelFactory.get_model('SVM')
        x_train_notext, y_train_notext = ImbalanceSampling.get_sampled_data('SMOTE', features_train_notext, y_train)
        model_notext_balanced.fit_model(x_train_notext, y_train_notext)

        model_text_balanced = ModelFactory.get_model('RandomForest')
        x_train_text, y_train_text = ImbalanceSampling.get_sampled_data('SMOTE', features_train_text, y_train)
        model_text_balanced.fit_model(x_train_text, y_train_text)

        pred1 = model_notext.predict_proba(features_test_notext)
        pred2 = model_text.predict_proba(features_test_text)
        pred3 = model_notext_balanced.predict_proba(features_test_notext)
        pred4 = model_text_balanced.predict_proba(features_test_text)

        final_pred = np.array([])
        for i in range(0, len(y_test)):
            first = (pred1[i][0] + 0.6*(pred2[i][0]))
            second = (pred1[i][1] + 0.6*(pred2[i][1]))
            val = 1
            if first > second:
                val = 0
            final_pred = np.append(final_pred, val)
        return ScoreMetrics.get_scores('ensemble', y_test, final_pred)

    @staticmethod
    def ensemble_averaging_all_features(features_train, features_test, y_train, y_test):
        # features_len_train = scale(features_train[0].reshape((features_train[0].shape[0], 1)))
        # features_len_test = scale(features_test[0].reshape((features_test[0].shape[0], 1)))
        # model = ModelFactory.get_model('SVM')
        # model.fit_model(features_len_train, y_train)
        # pred_len = model.predict_proba(features_len_test)
        #
        # features_stopwords_train = scale(features_train[1].reshape((features_train[1].shape[0], 1)))
        # features_stopwords_test = scale(features_test[1].reshape((features_test[1].shape[0], 1)))
        # model = ModelFactory.get_model('SVM')
        # model.fit_model(features_stopwords_train, y_train)
        # pred_stopwords = model.predict_proba(features_stopwords_test)
        #
        # features_similarity_train = scale(features_train[2].reshape((features_train[2].shape[0], 1)))
        # features_similarity_test = scale(features_test[2].reshape((features_test[2].shape[0], 1)))
        # model = ModelFactory.get_model('SVM')
        # model.fit_model(features_similarity_train, y_train)
        # pred_similarity = model.predict_proba(features_similarity_test)
        #
        # features_text_train = features_train[3]
        # features_text_test = features_test[3]
        # model = ModelFactory.get_model('RandomForest')
        # model.fit_model(features_text_train, y_train)
        # pred_text = model.predict_proba(features_text_test)
        #
        # features_functional_type_train = scale(features_train[4].reshape((features_train[4].shape[0], 1)))
        # features_functional_type_test = scale(features_test[4].reshape((features_test[4].shape[0], 1)))
        # model = ModelFactory.get_model('SVM')
        # model.fit_model(features_functional_type_train, y_train)
        # pred_functional_types = model.predict_proba(features_functional_type_test)

        ##BALANCED
        features_len_train = scale(features_train[0].reshape((features_train[0].shape[0], 1)))
        features_len_test = scale(features_test[0].reshape((features_test[0].shape[0], 1)))
        features_len_train, y_train_smote = ImbalanceSampling.get_sampled_data('SMOTE', features_len_train, y_train)
        model = ModelFactory.get_model('SVM')
        model.fit_model(features_len_train, y_train_smote)
        pred_len_smote = model.predict_proba(features_len_test)

        features_stopwords_train = scale(features_train[1].reshape((features_train[1].shape[0], 1)))
        features_stopwords_test = scale(features_test[1].reshape((features_test[1].shape[0], 1)))
        features_stopwords_train, y_train_smote = ImbalanceSampling.get_sampled_data('SMOTE', features_stopwords_train, y_train)
        model = ModelFactory.get_model('SVM')
        model.fit_model(features_stopwords_train, y_train_smote)
        pred_stopwords_smote = model.predict_proba(features_stopwords_test)

        features_similarity_train = scale(features_train[2].reshape((features_train[2].shape[0], 1)))
        features_similarity_test = scale(features_test[2].reshape((features_test[2].shape[0], 1)))
        features_similarity_train, y_train_smote = ImbalanceSampling.get_sampled_data('SMOTE', features_similarity_train, y_train)
        model = ModelFactory.get_model('SVM')
        model.fit_model(features_similarity_train, y_train_smote)
        pred_similarity_smote = model.predict_proba(features_similarity_test)

        features_text_train = features_train[3]
        features_text_test = features_test[3]
        features_text_train, y_train_smote = ImbalanceSampling.get_sampled_data('SMOTE', features_text_train, y_train)
        model = ModelFactory.get_model('RandomForest')
        model.fit_model(features_text_train, y_train_smote)
        pred_text_smote = model.predict_proba(features_text_test)

        features_functional_type_train = scale(features_train[4].reshape((features_train[4].shape[0], 1)))
        features_functional_type_test = scale(features_test[4].reshape((features_test[4].shape[0], 1)))
        features_functional_type_train, y_train_smote = ImbalanceSampling.get_sampled_data('SMOTE', features_functional_type_train, y_train)
        model = ModelFactory.get_model('SVM')
        model.fit_model(features_functional_type_train, y_train_smote)
        pred_functional_types_smote = model.predict_proba(features_functional_type_test)

        final_pred = np.array([])
        for i in range(0, len(y_test)):
            #first = 3.4 * pred_len[i][0] + 2.3 * pred_stopwords[i][0] + 1.3 * pred_similarity[i][0] + 2 * pred_text[i][0] + 1.2 * pred_functional_types[i][0]
            first = 0.1259028 * pred_len_smote[i][0] + 0.3 * pred_stopwords_smote[i][0] + 0.14755527 * pred_similarity_smote[i][0] + \
                      0.3 * pred_text_smote[i][0] + 0.16518774 * pred_functional_types_smote[i][0]
            #second = 3.4 * pred_len[i][1] + 2.3 * pred_stopwords[i][1] + 1.3 * pred_similarity[i][1] + 2 * pred_text[i][1] + 1.2 * pred_functional_types[i][1]
            second = 0.1259028 * pred_len_smote[i][1] + 0.3 * pred_stopwords_smote[i][1] + 0.14755527 * pred_similarity_smote[i][1] + \
                      0.3 * pred_text_smote[i][1] + 0.16518774 * pred_functional_types_smote[i][1]
            val = 1
            if first > second:
                val = 0
            final_pred = np.append(final_pred, val)
        return ScoreMetrics.get_scores('ensemble', y_test, final_pred)