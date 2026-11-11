"""Source-only nested development and immutable one-pass target orchestration."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, groups, load_source, read_json, sha256, verified, write_json
from .evaluation import (
    METHODS,
    SEARCHES,
    estimator,
    fit_predict,
    folds,
    metrics,
    positive_probability,
)
from .selection import DEPENDENCY_SHA, choose, search

SETTINGS = {
    "methods": list(METHODS),
    "population_size": 12,
    "budget": 90,
    "outer_seed": 2027,
    "inner_seed": 2028,
    "full_seeds": [0, 1, 2],
    "outer_search_seed_policy": "zero-based outer fold index (0,1,2), shared by all searches",
    "lr": {"solver": "liblinear", "C": 1.0, "max_iter": 2000, "random_state": 0},
    "threshold": 0.5,
    "smote": {"random_state": 2029, "k_neighbors": 5},
    "filter": {"score_func": "f_classif", "k": 10},
    "objective": {"mcc": 0.60, "ap": 0.40, "features": 0.02, "empty": 2.0},
    "undefined_objective_mcc": 0.0,
    "dependency_sha": DEPENDENCY_SHA,
    "final_candidate_policy": "highest mean outer MCC; ties by fewer features then method name",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args: str, root: Path = Path(".")) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def environment():
    packages = sorted({dist.metadata["Name"] for dist in importlib.metadata.distributions()})
    direct = importlib.metadata.distribution("metaheuristics-lab").read_text("direct_url.json")
    import json

    identity = json.loads(direct)
    if identity["vcs_info"]["commit_id"] != DEPENDENCY_SHA:
        raise ValueError("Optimizer installation is not the pinned VCS dependency")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "optimizer_direct_url": identity,
    }


def preflight(x, y):
    outer = folds(x, y, 2027)
    for train, _ in outer:
        folds(x[train], y[train], 2028)
    folds(x, y, 2028)
    return outer


def fold_record(train, valid):
    return {"train": train.tolist(), "valid": valid.tolist()}


def develop(raw_dir: Path, output: Path):
    if (output / "development.json").exists() or (output / "DEVELOPMENT_STARTED.json").exists():
        raise ValueError("Development output is immutable; choose a fresh directory")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Commit preflight implementation before scientific training")
    source = load_source(raw_dir)
    x, y = source.x, source.y
    outer = preflight(x, y)
    env = environment()
    write_json(
        output / "DEVELOPMENT_STARTED.json", {"time": now(), "commit": git("rev-parse", "HEAD")}
    )
    started = perf_counter()
    write_json(output / "settings.json", SETTINGS)
    write_json(output / "environment.json", env)
    write_json(output / "source_rows.json", {"y": y.tolist(), "groups": groups(x).tolist()})
    searches, outer_rows = [], []
    for index, (train, valid) in enumerate(outer):
        for method in METHODS:
            tick = perf_counter()
            mask = np.ones(20, dtype=int)
            if method in SEARCHES:
                result = search(method, x[train], y[train], seed=index)
                result.update(
                    {"stage": "outer", "outer_fold": index, "source_indices": train.tolist()}
                )
                searches.append(result)
                mask = np.array(result["mask"])
            probability, model = fit_predict(method, x, y, train, valid, mask)
            if method == "select_k_best":
                mask = model.named_steps["filter"].get_support().astype(int)
            outer_rows.append(
                {
                    "method": method,
                    "fold": index,
                    **fold_record(train, valid),
                    "mask": mask.tolist(),
                    "metrics": metrics(y[valid], probability),
                    "probabilities": probability.tolist(),
                    "classifier_fits": 1,
                    "seconds": perf_counter() - tick,
                }
            )
            print(f"outer {index} {method}: complete", flush=True)
    final = {}
    stability = {}
    for method in METHODS:
        if method in SEARCHES:
            repeats = []
            for seed in [0, 1, 2]:
                result = search(method, x, y, seed)
                result.update({"stage": "full", "source_indices": list(range(len(y)))})
                searches.append(result)
                repeats.append(result)
                print(f"full {method} seed {seed}: complete", flush=True)
            winner = choose(repeats)
            final[method] = {k: winner[k] for k in ("mask", "seed", "fitness")}
            pairs = []
            for a, b in combinations(repeats, 2):
                ma, mb = np.array(a["mask"], bool), np.array(b["mask"], bool)
                pairs.append(
                    {
                        "seeds": [a["seed"], b["seed"]],
                        "jaccard": float((ma & mb).sum() / (ma | mb).sum()),
                    }
                )
            stability[method] = pairs
        else:
            mask = np.ones(20, dtype=int)
            if method == "select_k_best":
                # Preprocessing only; no additional classifier fit.
                scaled = StandardScaler().fit_transform(x)
                mask = SelectKBest(f_classif, k=10).fit(scaled, y).get_support().astype(int)
            final[method] = {"mask": mask.tolist(), "seed": None, "fitness": None}
        final[method]["features"] = [
            name for name, bit in zip(FEATURES, final[method]["mask"], strict=True) if bit
        ]
    means = {
        m: float(np.mean([r["metrics"]["mcc"] or 0 for r in outer_rows if r["method"] == m]))
        for m in METHODS
    }
    candidate = min(METHODS, key=lambda m: (-means[m], sum(final[m]["mask"]), m))
    development = {
        "outer": outer_rows,
        "searches": searches,
        "final": final,
        "stability": stability,
        "candidate": candidate,
        "mean_outer_mcc": means,
        "objective_calls": sum(r["calls"] for r in searches),
        "objective_fits": sum(r["fits"] for r in searches),
        "outer_fits": len(outer_rows),
        "seconds": perf_counter() - started,
        "completed": now(),
    }
    assert development["objective_calls"] == 2160
    write_json(output / "development.json", development)
    return development


def protected_paths(root: Path, output: Path):
    """All scientific code, tests, contract and complete pre-target artifacts."""
    names = git("ls-files", root=root).splitlines()
    prefixes = ("src/", "tests/", "scripts/", "experiments/", ".github/")
    fixed = {
        "pyproject.toml",
        "AGENTS.md",
        "requirements.lock",
        "docs/PROJECT_SPEC.md",
        "docs/EXPERIMENT_PROTOCOL.md",
        "docs/DATASET_PROVENANCE.md",
        "docs/IMPLEMENTATION.md",
    }
    paths = [root / n for n in names if n.startswith(prefixes) or n in fixed]
    paths += [p for p in output.iterdir() if p.is_file() and p.name != "freeze.json"]
    return sorted(set(paths))


def freeze(root: Path, output: Path):
    path = output / "freeze.json"
    if path.exists():
        raise ValueError("Freeze is immutable")
    if git("status", "--porcelain", root=root):
        raise ValueError("Commit development artifacts before freeze")
    from .audit import audit_development

    audit_development(output)
    hashes = {
        str(p.relative_to(root)): sha256(p.read_bytes()) for p in protected_paths(root, output)
    }
    manifest = {
        "created": now(),
        "base_commit": git("rev-parse", "HEAD", root=root),
        "files": hashes,
        "settings": SETTINGS,
        "dataset": read_json(output / "dataset_manifest.json"),
        "final": read_json(output / "development.json")["final"],
    }
    write_json(path, manifest)
    return manifest


def verify_freeze(root: Path, output: Path, check_remote: bool = True):
    path = output / "freeze.json"
    manifest = read_json(path)
    relative = str(path.relative_to(root))
    freeze_sha = git("log", "-1", "--format=%H", "--", relative, root=root)
    if not freeze_sha:
        raise ValueError("Freeze is not committed")
    committed = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{freeze_sha}:{relative}"]
    )
    if committed != path.read_bytes():
        raise ValueError("Freeze manifest modified")
    if git("rev-parse", "HEAD", root=root) != freeze_sha:
        raise ValueError("Final scoring must run at the committed freeze HEAD")
    if git("status", "--porcelain", "--untracked-files=no", root=root):
        raise ValueError("Dirty tracked files at freeze")
    if manifest["settings"] != SETTINGS:
        raise ValueError("Settings differ from freeze")
    for name, expected in manifest["files"].items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError(f"Frozen identity mismatch: {name}")
    if check_remote:
        branch = git("branch", "--show-current", root=root)
        remote = git("ls-remote", "origin", f"refs/heads/{branch}", root=root).split()
        if not remote or remote[0] != freeze_sha:
            raise ValueError("Freeze HEAD has not been pushed")
    if environment() != read_json(output / "environment.json"):
        raise ValueError("Run environment differs from development")
    return freeze_sha


def start_final(output: Path, freeze_sha: str):
    path = output / "STARTED.json"
    # Exclusive creation remains after partial failure. Never delete to retry.
    with path.open("x") as stream:
        import json

        json.dump({"time": now(), "freeze_sha": freeze_sha}, stream, indent=2)
        stream.write("\n")


def final(raw_dir: Path, root: Path, output: Path):
    freeze_sha = verify_freeze(root, output)
    manifest = read_json(output / "freeze.json")
    for name, identity in manifest["dataset"]["files"].items():
        if sha256((raw_dir / name).read_bytes()) != identity["sha256"]:
            raise ValueError("Frozen raw data mismatch")
    start_final(output, freeze_sha)
    source = load_source(raw_dir)
    target = verified((raw_dir / "camel-1.6.csv").read_bytes(), "1.6")[0]
    seen = np.array([identifier in set(source.ids) for identifier in target.ids])
    prediction = {
        "row_index": list(range(len(target.y))),
        "y": target.y.tolist(),
        "seen_id": seen.tolist(),
        "probabilities": {},
    }
    rows, timings = {}, {}
    for method in METHODS:
        tick = perf_counter()
        mask = np.array(manifest["final"][method]["mask"], dtype=bool)
        model = estimator(method)
        # The filter is refit within the final full-source pipeline, and checked against freeze.
        columns = np.ones(20, bool) if method == "select_k_best" else mask
        model.fit(source.x[:, columns], source.y)
        fit_seconds = perf_counter() - tick
        if method == "select_k_best":
            assert np.array_equal(model.named_steps["filter"].get_support(), mask)
        tick = perf_counter()
        probability = positive_probability(model, target.x[:, columns])
        predict_seconds = perf_counter() - tick
        prediction["probabilities"][method] = probability.tolist()
        rows[method] = {}
        for subgroup, indices in {
            "whole": np.ones(len(seen), bool),
            "seen_id": seen,
            "unseen_id": ~seen,
        }.items():
            rows[method][subgroup] = {
                **metrics(target.y[indices], probability[indices]),
                "features": int(mask.sum()),
            }
        timings[method] = {
            "classifier_fits": 1,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    write_json(output / "target_predictions.json", prediction)
    write_json(output / "target_metrics.json", rows)
    write_json(output / "target_timing.json", timings)
    files = ["STARTED.json", "target_predictions.json", "target_metrics.json", "target_timing.json"]
    write_json(
        output / "COMPLETED.json",
        {
            "time": now(),
            "freeze_sha": freeze_sha,
            "freeze_digest": sha256((output / "freeze.json").read_bytes()),
            "classifier_fits": 8,
            "files": {name: sha256((output / name).read_bytes()) for name in files},
        },
    )
    return rows
