from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


class ImbalanceSampling:
    """
    Contains implementations of data sampling methods for tackling class imbalance problem,
    a chosen data imbalance sampling algorithm can be applied on data by calling
    ‘ImbalanceSampling.get_sampled_data(algo, x_train, y_train)’
    with ‘algo’ parameter being a string of values ‘SMOTE’, ‘ADASYN’, ‘RANDOM_OVERSAMPLE’, ‘RANDOM_UNDERSAMPLE’, ‘SMOTEEN’
    """

    @staticmethod
    def get_sampled_data(algo: str, x_train, y_train):
        """
        :param algo: algorithm string, possible values: 'SMOTE', 'ADASYN', 'RANDOM_UNDERSAMPLE', RANDOM_OVERSAMPLE'
        :param x_train:
        :param y_train:
        :return: returns data sampled with a specified imbalance samling algorithm
        """
        measure_function = ImbalanceSampling.get_sampling_algo(algo)
        return measure_function(x_train, y_train)

    @staticmethod
    def get_sampling_algo(format: str):
        if format == 'SMOTE':
            return ImbalanceSampling._smote
        if format == 'ADASYN':
            return ImbalanceSampling._adasyn
        if format == 'RANDOM_UNDERSAMPLE':
            return ImbalanceSampling._random_undersamplifier
        if format == 'RANDOM_OVERSAMPLE':
            return ImbalanceSampling._random_oversamplifier
        if format == 'SMOTEEN':
            return ImbalanceSampling._smoteen
        else:
            raise ValueError(format)

    @staticmethod
    def _smote(x_train, y_train):
        smote = SMOTE()
        return smote.fit_sample(x_train, y_train)

    @staticmethod
    def _adasyn(x_train, y_train):
        ada = ADASYN()
        return ada.fit_sample(x_train, y_train)

    @staticmethod
    def _random_undersamplifier(x_train, y_train):
        rus = RandomUnderSampler()
        return rus.fit_sample(x_train, y_train)

    @staticmethod
    def _random_oversamplifier(x_train, y_train):
        rus = RandomOverSampler()
        return rus.fit_sample(x_train, y_train)

    @staticmethod
    def _smoteen(x_train, y_train):
        smo = SMOTEENN()
        return smo.fit_sample(x_train, y_train)