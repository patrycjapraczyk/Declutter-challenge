from models.abstract_model import AbstractModel
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np


class AdaBoost(AbstractModel):
    def __init__(self, optimised=True):
        self.create_model(optimised)

    def create_model(self, optimised):
        #tuned hpyerparameters :(best parameters)  {'learning_rate': 0.1, 'n_estimators': 172, 'random_state': 0}
        if optimised:
            self.model = AdaBoostClassifier(learning_rate=0.1, n_estimators=170, random_state=0)
        else:
            self.model = AdaBoostClassifier()

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
