"""Entirely synthetic, offline exercise; never opens any data file."""

import numpy as np
from sklearn.datasets import make_classification

from .evaluation import METHODS, fit_predict, folds, metrics


def smoke():
    x, y = make_classification(n_samples=150, n_features=20, random_state=31)
    train, valid = folds(x, y, 2027)[0]
    result = {}
    for method in METHODS:
        mask = np.ones(20, dtype=int)
        if method.startswith("binary_"):
            from .selection import search

            mask = search(method, x[train], y[train], seed=0, budget=12)["mask"]
        probability, _ = fit_predict(method, x, y, train, valid, mask)
        result[method] = metrics(y[valid], probability)
    return result
