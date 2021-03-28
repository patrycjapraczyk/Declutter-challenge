from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
from models.model_exec import ModelExec
import numpy as np


# normalize a vector to have unit norm
def normalize(weights):
    result = norm(weights, 1)
    if result == 0.0:
        return weights
    return weights / result


# grid search weights
def grid_search():
    # define weights to consider
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    # iterate all possible combinations (cartesian product)
    for weights in product(w, repeat=2):
        # skip if all weights are equal
        if len(set(weights)) == 1:
            continue
        # hack, normalize weight vector
        weights = normalize(weights)
        # evaluate weights
        exec = ModelExec(include_comments=False, include_long_code=True)
        score = exec.kfold_split(10, weights[0], weights[1])
        score = score[score['name'] == 'ensemble']
        score = score['balanced_accuracy'].to_numpy()[0]
        score = np.mean(score)

        #score = evaluate_ensemble(members, weights, testX, testy)
        if score > best_score:
            best_score, best_weights = score, weights
            print('>%s %.3f' % (best_weights, best_score))
    print(best_score)
    return list(best_weights)

print(grid_search())