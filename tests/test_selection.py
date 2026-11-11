import importlib.metadata

import numpy as np
import pytest
from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch
from sklearn.datasets import make_classification

from software_defect_prediction.evaluation import SEARCHES, folds
from software_defect_prediction.selection import Objective, choose, fitness, optimizer, search


@pytest.mark.parametrize("method", SEARCHES)
def test_installed_engine_exact_calls_config_and_determinism(method):
    engine = optimizer(method)
    assert (
        type(engine)
        is {
            "binary_ga": BinaryGA,
            "binary_gwo_s": BinaryGWO,
            "binary_gwo_v": BinaryGWO,
            "binary_random_search": RandomSearch,
        }[method]
    )
    assert engine.population_size == 12
    if isinstance(engine, BinaryGWO):
        assert engine.transfer == method[-1]
    assert engine.domain == "binary"
    calls = []
    result = engine.optimize(
        Problem.binary(lambda mask: calls.append(mask) or float(mask.sum()), 20),
        max_evaluations=90,
        seed=0,
    )
    assert len(calls) == result.n_evaluations == len(result.history) == 90
    other = engine.optimize(
        Problem.binary(lambda m: float(m.sum()), 20), max_evaluations=90, seed=0
    )
    np.testing.assert_array_equal(result.history, other.history)
    assert "site-packages" in str(
        importlib.metadata.distribution("metaheuristics-lab").locate_file("")
    )


def test_empty_repeat_fits_and_formula():
    x, y = make_classification(n_samples=90, n_features=20, random_state=7)
    obj = Objective(x, y, folds(x, y, 2028))
    assert obj(np.zeros(20)) == 2.0
    assert obj.fits == 0
    value = obj(np.ones(20))
    assert obj(np.ones(20)) == value
    assert obj.fits == 6 and len(obj.trace) == 3
    assert fitness(0.2, 0.3, 10) == pytest.approx(0.53)
    assert obj.trace[1]["mask"] == obj.trace[2]["mask"]


def test_search_real_objective_and_ties():
    x, y = make_classification(n_samples=90, n_features=20, random_state=7)
    result = search("binary_ga", x, y, 1, budget=12)
    assert result["fits"] == 36
    assert choose([{**result, "seed": 2}, result])["seed"] == 1


def test_validation_predictors_cannot_change_fitted_scaler():
    from software_defect_prediction.evaluation import fit_predict

    x, y = make_classification(n_samples=90, n_features=20, random_state=7)
    train, valid = folds(x, y, 2028)[0]
    _, first = fit_predict("all_features_weighted", x, y, train, valid)
    x[valid] += 10000
    _, second = fit_predict("all_features_weighted", x, y, train, valid)
    np.testing.assert_array_equal(first.named_steps["lr"].coef_, second.named_steps["lr"].coef_)
