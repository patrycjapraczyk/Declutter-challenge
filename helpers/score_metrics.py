from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class ScoreMetrics:
    @staticmethod
    def print_scores(y_test, y_pred):
        print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
        print('Precision Score : ' + str(precision_score(y_test, y_pred)))
        print('Recall Score : ' + str(recall_score(y_test, y_pred)))
        print('F1 Score : ' + str(f1_score(y_test, y_pred)))