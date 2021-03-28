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
from models.dummy import Dummy


class ModelFactory:
    @staticmethod
    def get_model(format, optimised=True) -> AbstractModel:
        if format == 'LogisticRegression':
            return LogisticRegressionModel(optimised)
        if format == 'RandomForest':
            return RandomForestModel(optimised)
        if format == 'NaiveBayes':
            return NaiveBayes(optimised)
        if format == 'GradientBoosting':
            return GradientBoosting(optimised)
        if format == 'SVM':
            return SVM(optimised)
        if format == 'OneClassSVM':
            return OneClassSVMModel(optimised)
        if format == 'DecisionTree':
            return DecisionTree(optimised)
        if format == 'AdaBoost':
            return AdaBoost(optimised)
        if format == 'GaussianProcess':
            return GaussianProcess(optimised)
        if format == 'MLP':
            return MLP(optimised)
        if format == 'KNeighbors':
            return KNeighbors(optimised)
        if format == 'QuadraticDiscriminant':
            return QuadraticDiscriminant(optimised)
        if format == 'Dummy':
            return Dummy(optimised)
        else:
            raise ValueError(format)

    @staticmethod
    def get_models_list() -> list:
        list = ['LogisticRegression',
                'RandomForest', 'GradientBoosting', 'SVM',
                'DecisionTree', 'AdaBoost', 'NaiveBayes', 'GaussianProcess', 'MLP',
                'KNeighbors', 'QuadraticDiscriminant', 'Dummy']
        return list