from abc import ABC, abstractmethod


class AbstractTextRepresentation(ABC):
    @abstractmethod
    def create_model(self, corpus):
        pass

    @abstractmethod
    def vectorize(self, corpus):
        pass

    @abstractmethod
    def print(self):
        pass