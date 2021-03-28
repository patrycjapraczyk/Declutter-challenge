import pandas as pd
from models.model_exec import ModelExec

exec = ModelExec(include_comments=False, include_long_code=True)
from mlxtend.evaluate import paired_ttest_5x2cv
from statsmodels.stats.contingency_tables import mcnemar,cochrans_q
from models.model_factory import ModelFactory

# executes McNemar test to check the variance
# between results from k-fold validation of different models

model_names = ModelFactory.get_models_list()
result = exec.kfold_validate(2, 10)

data = []


for i in range(0, len(model_names)):
    name1 = model_names[i]
    df = pd.DataFrame()
    for j in range(0, len(model_names)):
        name2 = model_names[j]
        if name1 != name2:
            yes_yes = 0
            no_no = 0
            yes_no = 0
            no_yes = 0
            for index, row in result.iterrows():
                r1 = row[name1]
                r2 = row[name2]
                actual = row['actual']
                if r1 == actual and r2 == actual:
                    yes_yes += 1
                elif r1 != actual and r2 != actual:
                    no_no += 1
                elif r1 == actual and r2 != actual:
                    yes_no += 1
                elif r2 != actual and r2 == actual:
                    no_yes += 1
            conv_table = [[yes_yes, yes_no], [no_yes, no_no]]
            # calculate mcnemar test
            result_mcnemar = mcnemar(conv_table, exact=True)
            p = result_mcnemar.pvalue
            df[name2] = [p]
            statistic = result_mcnemar.statistic
            # summarize the finding
            print('class2 ', name1, 'class1', name2, 'statistic=%.3f, p-value=%.3f', p)
            # interpret the p-value
            alpha = 0.05
            if p > alpha:
                print('Same proportions of errors (fail to reject H0)')
            else:
                print('Different proportions of errors (reject H0)')
        df.to_csv(name1 + '_mcnemar.csv')