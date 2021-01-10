from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd


class ScoreMetrics:
    @staticmethod
    def print_scores(y_test, y_pred):
        print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
        print('Precision Score : ' + str(precision_score(y_test, y_pred)))
        print('Recall Score : ' + str(recall_score(y_test, y_pred)))
        print('F1 Score : ' + str(f1_score(y_test, y_pred)))

    @staticmethod
    def get_scores(name, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score_df = pd.DataFrame()
        score_df = score_df.append({'name': name, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, ignore_index=True)
        return score_df