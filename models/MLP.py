from models.abstract_model import AbstractModel
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np


class MLP(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(20,), learning_rate='adaptive', solver='adam')

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
