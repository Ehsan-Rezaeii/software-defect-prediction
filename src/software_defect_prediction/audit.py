"""Artifact-only independent audits. This module never fits an estimator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .data import EXPECTED, REVISION, read_json, sha256
from .evaluation import METHODS, SEARCHES


def independent_metrics(y, probability):
    """Use independent sklearn implementations to check the production formulas."""
    y, p = np.array(y), np.array(probability)
    assert y.ndim == p.ndim == 1 and len(y) == len(p) and len(y)
    assert np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pred = p >= 0.5
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    both = len(set(y)) == 2
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "n": len(y),
        "positive": tp + fn,
        "negative": tn + fp,
        "prevalence": float(y.mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mcc": float(matthews_corrcoef(y, pred)) if den else None,
        "f1": float(f1_score(y, pred)) if both and 2 * tp + fp + fn else None,
        "recall": float(recall_score(y, pred)) if both and tp + fn else None,
        "precision": float(precision_score(y, pred)) if both and tp + fp else None,
        "ap": float(average_precision_score(y, p)) if both else None,
        "predicted_positive_fraction": float(pred.mean()),
    }


def equal_metrics(actual, expected):
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            assert actual[key] is not None and np.isclose(
                actual[key], value, rtol=1e-12, atol=1e-12
            ), key


def validate_mask(mask):
    assert len(mask) == 20 and set(mask) <= {0, 1} and sum(mask) > 0


def audit_fold(train, valid, y, g):
    train, valid = np.array(train, int), np.array(valid, int)
    assert len(set(train)) == len(train) and len(set(valid)) == len(valid)
    assert set(train) | set(valid) == set(range(len(y)))
    assert not set(train) & set(valid)
    assert not set(g[train]) & set(g[valid])
    assert set(y[valid]) == set(y[train]) == {0, 1}
    assert np.bincount(y[train]).min() >= 6


def expected_folds(y, g, seed):
    return list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(
            np.zeros((len(y), 1)), y, g
        )
    )


def audit_development(output: Path):
    d = read_json(output / "development.json")
    source = read_json(output / "source_rows.json")
    y, g = np.array(source["y"]), np.array(source["groups"])
    assert len(y) == 872 and int(y.sum()) == 145 and len(set(g)) == 802
    assert len(d["outer"]) == 24 and len(d["searches"]) == 24
    outer = expected_folds(y, g, 2027)
    keys = {(m, f) for m in METHODS for f in range(3)}
    assert {(r["method"], r["fold"]) for r in d["outer"]} == keys
    for row in d["outer"]:
        train, valid = outer[row["fold"]]
        assert row["train"] == train.tolist() and row["valid"] == valid.tolist()
        audit_fold(train, valid, y, g)
        validate_mask(row["mask"])
        assert row["classifier_fits"] == 1
        equal_metrics(row["metrics"], independent_metrics(y[valid], row["probabilities"]))
    calls = fits = 0
    for method in SEARCHES:
        selected_runs = [r for r in d["searches"] if r["method"] == method]
        assert len(selected_runs) == 6
        assert sorted(r["seed"] for r in selected_runs if r["stage"] == "full") == [0, 1, 2]
        assert sorted(r["outer_fold"] for r in selected_runs if r["stage"] == "outer") == [0, 1, 2]
    for run in d["searches"]:
        assert run["population_size"] == 12
        assert run["budget"] == run["calls"] == len(run["trace"]) == len(run["history"]) == 90
        indices = np.array(run["source_indices"])
        if run["stage"] == "outer":
            assert indices.tolist() == outer[run["outer_fold"]][0].tolist()
            assert run["seed"] == run["outer_fold"]
        else:
            assert indices.tolist() == list(range(872))
        iy, ig = y[indices], g[indices]
        expected = expected_folds(iy, ig, 2028)
        for fold, (train, valid) in zip(run["inner_folds"], expected, strict=True):
            assert fold == {"train": train.tolist(), "valid": valid.tolist()}
            audit_fold(train, valid, iy, ig)
        best = float("inf")
        run_fits = 0
        for index, trace in enumerate(run["trace"]):
            assert trace["call"] == index + 1
            mask = trace["mask"]
            assert len(mask) == 20 and set(mask) <= {0, 1}
            assert trace["selected"] == sum(mask)
            if sum(mask):
                assert trace["fits"] == len(trace["fold_metrics"]) == 3
                for row, (_, valid) in zip(trace["fold_metrics"], expected, strict=True):
                    assert row["n"] == len(valid) and row["positive"] == int(iy[valid].sum())
                    assert row["tn"] + row["fp"] == row["negative"]
                    assert row["fn"] + row["tp"] == row["positive"]
                    assert row["objective_mcc"] == (row["mcc"] if row["mcc"] is not None else 0)
                mcc = np.mean([r["objective_mcc"] for r in trace["fold_metrics"]])
                ap = np.mean([r["ap"] for r in trace["fold_metrics"]])
                value = 0.6 * (1 - mcc) / 2 + 0.4 * (1 - ap) + 0.02 * sum(mask) / 20
            else:
                assert trace["fits"] == 0 and trace["fold_metrics"] == []
                value = 2.0
            assert np.isclose(trace["fitness"], value, rtol=1e-12, atol=1e-12)
            best = min(best, value)
            assert np.isclose(run["history"][index], best)
            run_fits += trace["fits"]
        assert run["fits"] == run_fits
        assert run["fitness"] == min(t["fitness"] for t in run["trace"])
        assert any(
            t["mask"] == run["mask"] and t["fitness"] == run["fitness"] for t in run["trace"]
        )
        calls += run["calls"]
        fits += run_fits
    assert d["objective_calls"] == calls == 2160
    assert d["objective_fits"] == fits and d["outer_fits"] == 24
    assert set(d["final"]) == set(METHODS)
    for method, frozen in d["final"].items():
        validate_mask(frozen["mask"])
        if method in SEARCHES:
            repeats = [r for r in d["searches"] if r["method"] == method and r["stage"] == "full"]
            winner = min(
                repeats, key=lambda r: (r["fitness"], sum(r["mask"]), r["seed"], tuple(r["mask"]))
            )
            assert all(frozen[k] == winner[k] for k in ["mask", "seed", "fitness"])
            for pair in d["stability"][method]:
                a, b = [
                    np.array(next(r["mask"] for r in repeats if r["seed"] == s), bool)
                    for s in pair["seeds"]
                ]
                assert np.isclose(pair["jaccard"], (a & b).sum() / (a | b).sum())
    manifest = read_json(output / "dataset_manifest.json")
    assert manifest["revision"] == REVISION
    for version, expected in EXPECTED.items():
        item = manifest["files"][f"camel-{version}.csv"]
        assert (
            item["git_blob"],
            item["bytes"],
            item["rows"],
            item["positive"],
            item["duplicate_groups"],
            item["excess_rows"],
            item["conflict_groups"],
        ) == expected
        assert len(item["sha256"]) == 64
    return {"objective_calls": calls, "objective_fits": fits, "outer_fits": 24}


def audit_results(root: Path, output: Path):
    counts = audit_development(output)
    freeze = read_json(output / "freeze.json")
    start, completed = [read_json(output / f"{name}.json") for name in ["STARTED", "COMPLETED"]]
    assert start["freeze_sha"] == completed["freeze_sha"]
    assert completed["freeze_digest"] == sha256((output / "freeze.json").read_bytes())
    assert (
        datetime.fromisoformat(freeze["created"])
        < datetime.fromisoformat(start["time"])
        < datetime.fromisoformat(completed["time"])
    )
    assert datetime.fromisoformat(
        read_json(output / "development.json")["completed"]
    ) < datetime.fromisoformat(freeze["created"])
    for name, digest in freeze["files"].items():
        assert sha256((root / name).read_bytes()) == digest, name
    for name, digest in completed["files"].items():
        assert sha256((output / name).read_bytes()) == digest, name
    assert freeze["final"] == read_json(output / "development.json")["final"]
    predictions = read_json(output / "target_predictions.json")
    y, seen = np.array(predictions["y"]), np.array(predictions["seen_id"], bool)
    assert predictions["row_index"] == list(range(965))
    assert len(y) == 965 and int(y.sum()) == 188 and int(seen.sum()) == 857
    actual = read_json(output / "target_metrics.json")
    assert set(predictions["probabilities"]) == set(actual) == set(METHODS)
    for method in METHODS:
        p = np.array(predictions["probabilities"][method])
        for subgroup, indices in {
            "whole": np.ones(965, bool),
            "seen_id": seen,
            "unseen_id": ~seen,
        }.items():
            equal_metrics(actual[method][subgroup], independent_metrics(y[indices], p[indices]))
            assert actual[method][subgroup]["features"] == sum(freeze["final"][method]["mask"])
    timing = read_json(output / "target_timing.json")
    assert set(timing) == set(METHODS)
    assert sum(v["classifier_fits"] for v in timing.values()) == completed["classifier_fits"] == 8
    for item in timing.values():
        assert item["fit_seconds"] > 0 and item["predict_seconds"] > 0
    return {**counts, "target_fits": 8, "target_rows": len(y), "freeze_sha": start["freeze_sha"]}
