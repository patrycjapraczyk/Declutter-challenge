from models.abstract_model import AbstractModel
from sklearn.dummy import DummyClassifier


class Dummy(AbstractModel):
    def __init__(self, optimised):
        self.create_model(optimised)

    def create_model(self, optimised):
        self.model = DummyClassifier(strategy="most_frequent")

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def get_model(self):
        return self.model

    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred

    def print(self):
        pass
