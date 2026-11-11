"""Thin adapters to the pinned Project 1 optimizers; no fitness cache."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from metaheuristics_lab import BinaryGA, BinaryGWO, Problem, RandomSearch

from .evaluation import fit_predict, folds, metrics

DEPENDENCY_SHA = "6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e"


def optimizer(method: str):
    factories = {
        "binary_ga": lambda: BinaryGA(population_size=12),
        "binary_gwo_s": lambda: BinaryGWO(population_size=12, transfer="s"),
        "binary_gwo_v": lambda: BinaryGWO(population_size=12, transfer="v"),
        "binary_random_search": lambda: RandomSearch(population_size=12, domain="binary"),
    }
    return factories[method]()


def fitness(mcc: float, ap: float, selected: int) -> float:
    return 0.60 * (1 - mcc) / 2 + 0.40 * (1 - ap) + 0.02 * selected / 20


class Objective:
    """Every invocation records evidence, including duplicate and empty masks."""

    def __init__(self, x, y, splits):
        self.x, self.y, self.splits = x, y, splits
        self.trace = []
        self.fits = 0

    def __call__(self, candidate):
        started = perf_counter()
        mask = np.asarray(candidate, dtype=int)
        if mask.shape != (20,) or not np.isin(mask, [0, 1]).all():
            raise ValueError("Expected 20 binary decisions")
        selected = int(mask.sum())
        rows = []
        value = 2.0
        if selected:
            for train, valid in self.splits:
                probability, _ = fit_predict(
                    "all_features_weighted", self.x, self.y, train, valid, mask
                )
                self.fits += 1
                row = metrics(self.y[valid], probability)
                # sklearn defines zero-denominator MCC as zero; objective uses that convention.
                row["objective_mcc"] = row["mcc"] if row["mcc"] is not None else 0.0
                if row["ap"] is None:
                    raise ValueError("Inner AP undefined")
                rows.append(row)
            value = fitness(
                float(np.mean([r["objective_mcc"] for r in rows])),
                float(np.mean([r["ap"] for r in rows])),
                selected,
            )
        self.trace.append(
            {
                "call": len(self.trace) + 1,
                "mask": mask.tolist(),
                "selected": selected,
                "fitness": value,
                "fold_metrics": rows,
                "fits": len(rows),
                "seconds": perf_counter() - started,
            }
        )
        return value


def search(method, x, y, seed, budget=90):
    splits = folds(x, y, 2028)
    objective = Objective(x, y, splits)
    engine = optimizer(method)
    result = engine.optimize(Problem.binary(objective, 20), max_evaluations=budget, seed=seed)
    assert result.n_evaluations == len(objective.trace) == budget
    return {
        "method": method,
        "seed": seed,
        "population_size": engine.population_size,
        "budget": budget,
        "calls": len(objective.trace),
        "fits": objective.fits,
        "mask": result.best_solution.astype(int).tolist(),
        "fitness": result.best_fitness,
        "history": result.history.tolist(),
        "seconds": result.runtime_seconds,
        "trace": objective.trace,
        "inner_folds": [{"train": a.tolist(), "valid": b.tolist()} for a, b in splits],
    }


def choose(repeats):
    return min(repeats, key=lambda r: (r["fitness"], sum(r["mask"]), r["seed"], tuple(r["mask"])))
