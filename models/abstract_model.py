from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def fit_model(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def print(self):
        pass