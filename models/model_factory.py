from models.abstract_model import AbstractModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.naive_bayes import NaiveBayes
from models.gradient_boosting import GradientBoosting
from models.decision_tree_classifier import DecisionTree
from models.ada_boost import AdaBoost
from models.SVM import SVM
from models.one_class_SVM import OneClassSVMModel
from models.gaussian_process import GaussianProcess
from models.MLP import MLP
from models.quadratic_discriminant_analysis import QuadraticDiscriminant
from models.K_neighbors_classifier import KNeighbors


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
        if format == 'OneClassSVM':
            return OneClassSVMModel()
        if format == 'DecisionTree':
            return DecisionTree()
        if format == 'AdaBoost':
            return AdaBoost()
        if format == 'GaussianProcess':
            return GaussianProcess()
        if format == 'MLP':
            return MLP()
        if format == 'KNeighbors':
            return KNeighbors()
        if format == 'QuadraticDiscriminant':
            return QuadraticDiscriminant()
        else:
            raise ValueError(format)

    @staticmethod
    def get_models_list() -> list:
        list = ['LogisticRegression',
                'RandomForest', 'GradientBoosting', 'SVM', 'OneClassSVM',
                'DecisionTree', 'AdaBoost', 'NaiveBayes', 'GaussianProcess', 'MLP',
                'KNeighbors', 'QuadraticDiscriminant']
        return list