from models.abstract_model import AbstractModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(AbstractModel):
    def __init__(self, optimised):
        self.create_model(optimised)

    def create_model(self, optimised):
        self.model = GaussianNB()

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
