from models.abstract_model import AbstractModel
from sklearn.svm import SVC


class SVM(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = SVC(C=1, random_state=42, max_iter=2000, probability=True)

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
