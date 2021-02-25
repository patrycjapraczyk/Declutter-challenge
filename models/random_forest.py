from models.abstract_model import AbstractModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


class RandomForestModel(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        #'bootstrap': True, 'max_depth': 80, 'max_features': 2, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 100
        self.model = RandomForestClassifier(bootstrap=True, max_depth=90, max_features=2, min_samples_leaf=5,
                                            min_samples_split=8, n_estimators=200, random_state=42)

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
        col = ["Comment Length", "Stopwords number", "Comment/code similarity"]
        y = self.model.feature_importances_
        # plot
        fig, ax = plt.subplots()
        width = 0.4  # the width of the bars
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, width, color="green")
        ax.set_yticks(ind + width / 10)
        ax.set_yticklabels(col, minor=False)

        plt.title("Feature importance in RandomForest Classifier")
        plt.xlabel("Relative importance")
        plt.ylabel("feature")
        plt.figure(figsize=(5, 5))
        fig.set_size_inches(6.5, 4.5, forward=True)
