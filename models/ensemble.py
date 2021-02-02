from models.model_factory import ModelFactory
import numpy as np
from statistics import mode
from helpers.score_metrics import ScoreMetrics


class Ensemble:
    @staticmethod
    def get_ensemble_score(mode, features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test):
        if mode == 'MAX_VOTE':
            Ensemble.ensemble_max_vote(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test)
        elif mode == 'AVERAGING':
            Ensemble.ensemble_averaging(features_train_text, features_train_notext,
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
        ScoreMetrics.print_scores(y_test, final_pred)

    @staticmethod
    def ensemble_averaging(features_train_text, features_train_notext,
                     features_test_text, features_test_notext, y_train, y_test):
        model_notext = ModelFactory.get_model('LogisticRegression')
        model_text = ModelFactory.get_model('RandomForest')

        model_notext.fit_model(features_train_notext, y_train)
        model_text.fit_model(features_train_text, y_train)

        pred1 = model_notext.predict_proba(features_test_notext)
        pred2 = model_text.predict_proba(features_test_text)

        final_pred = np.array([])
        for i in range(0, len(y_test)):
            first = (7*pred1[i][0] + 3*pred2[i][0])/10
            second = (7*pred1[i][1] + 3*pred2[i][1])/10
            val = 1
            if first > second:
                val = 0
            final_pred = np.append(final_pred, val)
        ScoreMetrics.print_scores(y_test, final_pred)