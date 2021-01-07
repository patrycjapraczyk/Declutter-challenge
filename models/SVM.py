from models.abstract_model import AbstractModel
from sklearn.svm import LinearSVC


class SVM(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = LinearSVC(penalty='l2', C=1, random_state=42)

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred
