from models.model_factory import ModelFactory
import numpy as np
from statistics import mode
from helpers.score_metrics import ScoreMetrics
from helpers.class_imbalance_sampling import ImbalanceSampling
from sklearn.preprocessing import LabelEncoder, normalize, scale
from itertools import product



class Ensemble:
    @staticmethod
    def get_ensemble_score(mode, features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test, w1, w2, imbalance_sampling=None):
        if mode == 'MAX_VOTE':
            return Ensemble.ensemble_max_vote(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test, imbalance_sampling)
        elif mode == 'AVERAGING':
            return Ensemble.ensemble_averaging(features_train_text, features_train_notext,
                                       features_test_text, features_test_notext, y_train, y_test, w1, w2, imbalance_sampling)

    @staticmethod
    def ensemble_max_vote(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test, imbalance_sampling=None):
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
                     features_test_text, features_test_notext, y_train, y_test, w1, w2, imbalance_sampling=None):
        model_notext = ModelFactory.get_model('SVM')
        model_text = ModelFactory.get_model('RandomForest', optimised=False)
        y_train_text = y_train

        if imbalance_sampling != None:
            features_train_text, y_train_text = ImbalanceSampling.get_sampled_data(imbalance_sampling, features_train_text,
                                                                             y_train)
            features_train_notext, y_train = ImbalanceSampling.get_sampled_data(imbalance_sampling, features_train_notext, y_train)

            # models that perform best with imbalance sampling
            model_notext = ModelFactory.get_model('MLP', optimised=True)
            model_text = ModelFactory.get_model('SVM', optimised=False)

        model_notext.fit_model(features_train_notext, y_train)
        model_text.fit_model(features_train_text, y_train_text)

        pred1 = model_notext.predict_proba(features_test_notext)
        pred2 = model_text.predict_proba(features_test_text)

        final_pred = np.array([])
        for i in range(0, len(y_test)):
            first = (w1*pred1[i][0] + w2*(pred2[i][0]))
            second = (w1*pred1[i][1] + w2*(pred2[i][1]))
            val = 1
            if first > second:
                val = 0
            final_pred = np.append(final_pred, val)
        return ScoreMetrics.get_scores('ensemble', y_test, final_pred)