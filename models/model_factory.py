from models.abstract_model import AbstractModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.naive_bayes import NaiveBayes
from models.gradient_boosting import GradientBoosting
from models.SVM import SVM


class ModelFactory:
    @staticmethod
    def get_model(format) -> AbstractModel:
        if format == 'LogisticRegression':
            return LogisticRegressionModel()
        if format == 'RandomForest':
            return RandomForestModel()
        if format == 'NaiveBayes':
            return NaiveBayes()
        if format == 'GradientBoosting':
            return GradientBoosting()
        if format == 'SVM':
            return SVM()
        else:
            raise ValueError(format)

    @staticmethod
    def get_models_list() -> list:
        list = ['LogisticRegression', 'RandomForest', 'NaiveBayes', 'GradientBoosting', 'SVM']
        return list