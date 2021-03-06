from models.abstract_model import AbstractModel
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel(AbstractModel):
    def __init__(self, optimised=True):
        self.create_model(optimised)

    def create_model(self, optimised):
        if optimised:
            self.model = LogisticRegression(C=1.7575106248547894, penalty='l2')
        else:
            self.model = LogisticRegression()

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def print(self):
        pass
