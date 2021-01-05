from abc import ABC, abstractmethod


class AbstractTextRepresentation(ABC):
    @abstractmethod
    def vectorize(self):
        pass

    @abstractmethod
    def print(self):
        pass