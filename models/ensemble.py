from models.model_factory import ModelFactory
import numpy as np
from statistics import mode
from helpers.score_metrics import ScoreMetrics
from helpers.class_imbalance_sampling import ImbalanceSampling


class Ensemble:
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
            first = (pred1[i][0] + pred3[i][0] + 0.6*(pred2[i][0] + pred4[i][0]))
            second = (pred1[i][1] + pred3[i][1] + 0.6*(pred2[i][1] + pred4[i][1]))
            val = 1
            if first > second:
                val = 0
            final_pred = np.append(final_pred, val)
        return ScoreMetrics.get_scores('ensemble', y_test, final_pred)