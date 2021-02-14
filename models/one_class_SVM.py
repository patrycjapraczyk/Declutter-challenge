from models.abstract_model import AbstractModel
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class OneClassSVMModel(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = OneClassSVM(gamma='auto')

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train)

    def predict(self, x_test):
        # train_normal = x_test[x_test['y'] == 0]
        # train_outliers = x_test[x_test['y'] == 1]
        # outlier_prop = len(train_outliers) / len(train_normal)
        y_pred = self.model.predict(x_test)
        #score = f1_score(y_test, y_pred)
        #print('F1 Score: %.3f' % score)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        return y_pred

    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def print(self):
        pass
