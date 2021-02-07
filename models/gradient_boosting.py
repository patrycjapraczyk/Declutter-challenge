from models.abstract_model import AbstractModel
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoosting(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=600, min_samples_leaf=20, n_estimators=660, max_depth=5, subsample=0.7, warm_start=True)

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
