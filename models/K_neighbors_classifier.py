from models.abstract_model import AbstractModel
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


class KNeighbors(AbstractModel):
    def __init__(self, optimised=True):
        self.create_model(optimised)

    def create_model(self, optimised):
        if optimised:
            self.model = KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=14, p=2, weights='uniform')
        else:
            self.model = KNeighborsClassifier()

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
