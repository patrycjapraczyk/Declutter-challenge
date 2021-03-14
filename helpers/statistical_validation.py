#%%

# compare logistic regression and lda for binary classification
# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from matplotlib import pyplot
from models.model_exec import ModelExec
import pandas as pd

# # define dataset
# # evaluate model 1
# model1 = LogisticRegression()
# cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores1 = cross_val_score(model1, X, y, scoring='accuracy', cv=cv1, n_jobs=-1)
# print('LogisticRegression Mean Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
# # evaluate model 2
# model2 = LinearDiscriminantAnalysis()
# cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv2, n_jobs=-1)
# print('LinearDiscriminantAnalysis Mean Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))
# # plot the results
# pyplot.boxplot([scores1, scores2], labels=['LR', 'LDA'], showmeans=True)
# pyplot.show()

from mlxtend.evaluate import paired_ttest_5x2cv
from models.model_factory import ModelFactory


class StatisticalValidation:
    @staticmethod
    def get_validation_measure(format):
        if format == 'T-TEST':
            return StatisticalValidation._t_test

    @staticmethod
    def _t_test(model1, model2, X, y):
        return paired_ttest_5x2cv(estimator1=model1, estimator2=model2, X=X, y=y, scoring='f1', random_seed=1)


def exec_paired_test():
    exec = ModelExec(include_comments=False, include_long_code=True)
    features = exec.extract_features(exec.data)
    X = exec.combine_features(features, include_comments=False)
    y = exec.data['non-information']
    model_names = ModelFactory.get_models_list()
    df = pd.DataFrame()

    for i in range(0, len(model_names)):
        name = model_names[i]
        model1 = ModelFactory.get_model(name)
        model1 = model1.model
        score_df = pd.DataFrame()
        score_df = score_df.append({'name': name}, ignore_index=True)
        for j in range(0, len(model_names)):
            name2 = model_names[j]
            score_df = pd.DataFrame(columns=['name'] + model_names)
            if name == name2:
                continue
            model2 = ModelFactory.get_model(name2)
            model2 = model2.model
            measure_function = StatisticalValidation.get_validation_measure('T-TEST')
            # check if difference between algorithms is real
            t, p = measure_function(model1, model2, X, y)
            score_df = pd.DataFrame(columns=['name'] + model_names)
            score_df[name] = p
            # summarize
            # print('P-value: %.3f, t-Statistic: %.3f' % (p, t))
            # # interpret the result
            # if p <= 0.05:
            #     print('Difference between mean performance is probably real between ' + name + ' and ' + name2)
            # else:
            #     print('Algorithms probably have the same performance ' + name + ' and ' + name2)
        df.append(score_df)
        i = 0
    print(df)

exec_paired_test()