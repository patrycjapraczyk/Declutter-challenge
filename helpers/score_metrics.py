from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


class ScoreMetrics:
    @staticmethod
    def print_scores(y_test, y_pred):
        print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
        print('Precision Score : ' + str(precision_score(y_test, y_pred)))
        print('Recall Score : ' + str(recall_score(y_test, y_pred)))
        print('F1 Score : ' + str(f1_score(y_test, y_pred)))
        print('Matthews_corrcoef: ' + str(matthews_corrcoef(y_test, y_pred)))
        print('Balanced accuracy score: ' + str(balanced_accuracy_score(y_test, y_pred)))
        print('Confusion matrix: \n' + str(confusion_matrix(y_test, y_pred)))

    @staticmethod
    def get_scores(name: str, y_test, y_pred) -> pd.DataFrame:
        """
        :param name: the name of the classifier tested
        :param y_test
        :param y_pred
        :return: a dataframe with various score metrics
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, labels=np.unique(y_pred))
        matthews_coeff = matthews_corrcoef(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        score_df = pd.DataFrame()
        score_df = score_df.append({'name': name, 'accuracy': accuracy, 'precision': precision,
                                    'recall': recall, 'f1': f1, 'matthews_corrcoef': matthews_coeff,
                                    'balanced_accuracy': balanced_accuracy, 'confusion_matrix': conf_mat}, ignore_index=True)
        return score_df