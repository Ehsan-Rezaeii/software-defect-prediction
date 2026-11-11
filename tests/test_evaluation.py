import numpy as np
import pytest
from sklearn.datasets import make_classification

from software_defect_prediction.data import groups
from software_defect_prediction.evaluation import (
    METHODS,
    estimator,
    fit_predict,
    folds,
    metrics,
    positive_probability,
)


@pytest.fixture
def synthetic():
    return make_classification(n_samples=150, n_features=20, weights=[0.7, 0.3], random_state=42)


def test_metrics_confusion_ap_and_undefined():
    actual = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.4, 0.9]))
    assert [actual[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]
    assert actual["mcc"] == 0
    assert actual["f1"] == 0.5
    assert actual["ap"] == pytest.approx(5 / 6)
    zero = metrics(np.zeros(3), np.zeros(3))
    assert all(zero[k] is None for k in ["mcc", "f1", "recall", "precision", "ap"])
    assert metrics(np.ones(3), np.ones(3))["ap"] is None
    with pytest.raises(ValueError):
        metrics(np.array([1]), np.array([np.nan]))


def test_positive_probability_column():
    class Reversed:
        classes_ = [1, 0]

        def predict_proba(self, x):
            return np.array([[0.8, 0.2]])

    assert positive_probability(Reversed(), [[0]]).tolist() == [0.8]


@pytest.mark.parametrize("method", METHODS)
def test_fold_local_fixed_parameters_and_label_injection(synthetic, method):
    x, y = synthetic
    train, valid = folds(x, y, 2028)[0]
    probability, model = fit_predict(method, x, y, train, valid)
    np.testing.assert_allclose(model.named_steps["scale"].mean_, x[train].mean(axis=0))
    lr = model.named_steps["lr"]
    assert (lr.solver, lr.C, lr.max_iter, lr.random_state) == ("liblinear", 1.0, 2000, 0)
    corrupted = y.copy()
    corrupted[valid] = 1 - corrupted[valid]
    other, _ = fit_predict(method, x, corrupted, train, valid)
    np.testing.assert_array_equal(probability, other)
    if method == "all_features_smote":
        smote = model.named_steps["smote"]
        assert (smote.random_state, smote.k_neighbors) == (2029, 5)


def test_duplicate_group_disjointness(synthetic):
    x, y = synthetic
    x, y = np.concatenate([x, x[:10]]), np.concatenate([y, y[:10]])
    g = groups(x)
    for train, valid in folds(x, y, 2027):
        assert not set(g[train]) & set(g[valid])
        for a, b in folds(x[train], y[train], 2028):
            assert not set(g[train[a]]) & set(g[train[b]])
    assert not hasattr(estimator("select_k_best").named_steps["filter"], "scores_")
