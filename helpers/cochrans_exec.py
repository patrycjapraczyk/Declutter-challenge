import pandas as pd
from models.model_exec import ModelExec

exec = ModelExec(include_comments=False, include_long_code=True)
from mlxtend.evaluate import paired_ttest_5x2cv
from statsmodels.stats.contingency_tables import mcnemar,cochrans_q
from models.model_factory import ModelFactory


model_names = ModelFactory.get_models_list()
exec = ModelExec(include_comments=False, include_long_code=True, comment_vectoriser='BOW')
result_bow = exec.kfold_validate(2, 10)

exec = ModelExec(include_comments=False, include_long_code=True, comment_vectoriser='B-NGRAM')
result_ngram = exec.kfold_validate(2, 10)

exec = ModelExec(include_comments=False, include_long_code=True, comment_vectoriser='TFIDF')
result_tfidf = exec.kfold_validate(2, 10)

exec = ModelExec(include_comments=False, include_long_code=True, comment_vectoriser='W2V')
result_w2v = exec.kfold_validate(2, 10)

data = []
for i in range(0, len(model_names)):
    name = model_names[i]
    data.append(result_bow[name].tolist())
    data.append(result_ngram[name].tolist())
    data.append(result_tfidf[name].tolist())
    data.append(result_w2v[name].tolist())

res = cochrans_q(data, return_object=True)
print(res.statistic, res.pvalue)