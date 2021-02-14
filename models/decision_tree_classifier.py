from models.abstract_model import AbstractModel
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


class DecisionTree(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.01)

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
