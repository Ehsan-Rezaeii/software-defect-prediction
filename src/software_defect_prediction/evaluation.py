"""Fold-local estimators, positive-class probabilities and explicit metrics."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import groups

METHODS = (
    "all_features",
    "all_features_weighted",
    "all_features_smote",
    "select_k_best",
    "binary_ga",
    "binary_gwo_s",
    "binary_gwo_v",
    "binary_random_search",
)
SEARCHES = METHODS[4:]


def estimator(method: str):
    if method not in METHODS:
        raise ValueError(method)
    weighted = method not in ("all_features", "all_features_smote")
    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=2000,
        random_state=0,
        class_weight="balanced" if weighted else None,
    )
    steps = [("scale", StandardScaler())]
    if method == "select_k_best":
        steps.append(("filter", SelectKBest(f_classif, k=10)))
    if method == "all_features_smote":
        steps.append(("smote", SMOTE(random_state=2029, k_neighbors=5)))
    steps.append(("lr", lr))
    return ImbPipeline(steps) if method == "all_features_smote" else Pipeline(steps)


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError("Missing unique positive class 1")
    return model.predict_proba(x)[:, indices[0]]


def metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    if y.shape != probability.shape or y.ndim != 1 or not len(y):
        raise ValueError("Nonempty one-dimensional matching vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid label or probability")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probability range")
    predicted = probability >= 0.5
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    both = len(set(y)) == 2
    return {
        "n": len(y),
        "positive": tp + fn,
        "negative": tn + fp,
        "prevalence": float(y.mean()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mcc": (tp * tn - fp * fn) / den if den else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if both and 2 * tp + fp + fn else None,
        "recall": tp / (tp + fn) if both and tp + fn else None,
        "precision": tp / (tp + fp) if both and tp + fp else None,
        "ap": float(average_precision_score(y, probability)) if len(set(y)) == 2 else None,
        "predicted_positive_fraction": float(predicted.mean()),
    }


def folds(x: np.ndarray, y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    g = groups(x)
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y, g))
    for train, valid in splits:
        assert not set(g[train]) & set(g[valid]), "Duplicate-vector leakage"
        for indices in (train, valid):
            if set(y[indices]) != {0, 1}:
                raise ValueError("Both classes required in every fold")
        if min(np.bincount(y[train])) < 6:
            raise ValueError("SMOTE requires at least six minority training rows")
    return splits


def fit_predict(method: str, x, y, train, valid, mask=None):
    """Only training indices enter fit; validation labels are never an argument."""
    columns = np.ones(x.shape[1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not columns.any():
        raise ValueError("Cannot fit an empty mask")
    model = estimator(method)
    model.fit(x[train][:, columns], y[train])
    return positive_probability(model, x[valid][:, columns]), model
