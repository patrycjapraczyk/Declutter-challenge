from models.abstract_model import AbstractModel
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes(AbstractModel):
    def __init__(self):
        self.create_model()

    def create_model(self):
        self.model = MultinomialNB()

    def fit_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred
