import pandas as pd
from models.model_exec import ModelExec

exec = ModelExec(include_comments=False, include_long_code=True)
from mlxtend.evaluate import paired_ttest_5x2cv
from statsmodels.stats.contingency_tables import mcnemar,cochrans_q
from models.model_factory import ModelFactory


model_names = ModelFactory.get_models_list()
result = exec.kfold_validate(2, 10)

data = []
for i in range(0, len(model_names)):
    name = model_names[i]
    data.append(result[name].tolist())

res = cochrans_q(data, return_object=True)
print(res.statistic, res.pvalue)