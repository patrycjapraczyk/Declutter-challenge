from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


class ImbalanceSampling:
    @staticmethod
    def get_sampled_data(algo, x_train, y_train):
        measure_function = ImbalanceSampling.get_sampling_algo(algo)
        return measure_function(x_train, y_train)

    @staticmethod
    def get_sampling_algo(format):
        if format == 'SMOTE':
            return ImbalanceSampling._smote
        if format == 'ADASYN':
            return ImbalanceSampling._adasyn
        if format == 'RANDOM_UNDERSAMPLE':
            return ImbalanceSampling._random_undersamplifier
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
    def _smoteen(x_train, y_train):
        smo = SMOTEENN()
        return smo.fit_sample(x_train, y_train)