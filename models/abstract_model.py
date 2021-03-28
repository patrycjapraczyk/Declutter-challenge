from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def create_model(self, optimised=False):
        pass

    @abstractmethod
    def fit_model(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def predict_proba(self, x_test):
        pass

    @abstractmethod
    def print(self):
        pass

