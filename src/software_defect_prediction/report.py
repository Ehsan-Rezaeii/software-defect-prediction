"""Render reports and figures only from saved artifacts, without model fitting."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from .audit import audit_results
from .data import FEATURES, read_json, sha256, write_json
from .evaluation import METHODS, SEARCHES


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def report(root: Path, output: Path):
    if (output / "artifact_manifest.json").exists():
        raise ValueError("Audited experiment output is immutable")
    audited = audit_results(root, output)
    development = read_json(output / "development.json")
    scores = read_json(output / "target_metrics.json")
    predictions = read_json(output / "target_predictions.json")
    timing = read_json(output / "target_timing.json")
    final = development["final"]
    candidate = development["candidate"]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def save(name):
        plt.tight_layout()
        plt.savefig(figures / name, dpi=160, bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, label in zip(axes, ["mcc", "ap"], ["MCC", "Average Precision"], strict=True):
        values = [scores[m]["whole"][metric] for m in METHODS]
        ax.barh(
            METHODS, values, color=["#356c9a" if m == candidate else "#86a9bd" for m in METHODS]
        )
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title("Frozen Camel 1.4 → 1.6; all 965 target rows")
    save("target_comparison.png")

    plt.figure(figsize=(9, 4))
    for method in SEARCHES:
        runs = [
            r for r in development["searches"] if r["method"] == method and r["stage"] == "full"
        ]
        histories = np.array([r["history"] for r in runs])
        plt.plot(range(1, 91), histories.mean(axis=0), label=method)
        plt.fill_between(range(1, 91), histories.min(axis=0), histories.max(axis=0), alpha=0.1)
    plt.xlabel("Objective calls (population 12; no cache)")
    plt.ylabel("Best inner fitness; lower is better")
    plt.title("Full-development searches: mean and range across three seeds")
    plt.legend()
    save("fitness_traces.png")

    plt.figure(figsize=(11, 4))
    plt.imshow([final[m]["mask"] for m in METHODS], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(20), FEATURES, rotation=60)
    plt.yticks(range(8), METHODS)
    plt.title("Frozen masks; chosen using Camel 1.4 only")
    save("frozen_masks.png")

    plt.figure(figsize=(8, 5))
    for method in METHODS:
        precision, recall, _ = precision_recall_curve(
            predictions["y"], predictions["probabilities"][method]
        )
        plt.plot(recall, precision, label=method)
    plt.axhline(np.mean(predictions["y"]), color="gray", linestyle="--", label="Target prevalence")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Target PR curves from the same saved probability vectors")
    plt.legend(fontsize=8)
    save("precision_recall.png")

    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, method in zip(axes.flat, METHODS, strict=True):
        row = scores[method]["whole"]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=777)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=method)
    save("confusion_matrices.png")

    plt.figure(figsize=(9, 4))
    costs = [
        sum(r["seconds"] for r in development["searches"] if r["method"] == m) for m in METHODS
    ]
    plt.barh(METHODS, costs, color="#356c9a")
    plt.xlabel("Search wall seconds (outer and full-development searches)")
    plt.title("Search overhead; baseline final training time is recorded separately")
    save("search_cost.png")

    fields = [
        "method",
        "features",
        "mcc",
        "f1",
        "recall",
        "precision",
        "ap",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with (output / "target_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {"method": method, **{k: scores[method]["whole"][k] for k in fields[1:]}}
            )
    table = [
        "| Method | Features | MCC | F1 | Recall | Precision | AP | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row = scores[method]["whole"]
        table.append(
            f"| {method} | {row['features']} | "
            + " | ".join(number(row[k]) for k in ["mcc", "f1", "recall", "precision", "ap"])
            + " | "
            + "/".join(str(row[k]) for k in ["tn", "fp", "fn", "tp"])
            + " |"
        )
    table_text = "\n".join(table)
    outer_table = ["| Method | MCC mean ± sample SD | AP mean ± sample SD |", "|---|---:|---:|"]
    for method in METHODS:
        vals = [
            [r["metrics"][k] or 0 for r in development["outer"] if r["method"] == method]
            for k in ["mcc", "ap"]
        ]
        outer_table.append(
            f"| {method} | "
            + " | ".join(f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" for v in vals)
            + " |"
        )
    subgroup = [
        "| Method | Subgroup | n | Positive | MCC | F1 | Recall | Precision | AP | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        for name in ["seen_id", "unseen_id"]:
            row = scores[method][name]
            subgroup.append(
                f"| {method} | {name} | {row['n']} | {row['positive']} | "
                + " | ".join(number(row[k]) for k in ["mcc", "f1", "recall", "precision", "ap"])
                + " | "
                + "/".join(str(row[k]) for k in ["tn", "fp", "fn", "tp"])
                + " |"
            )
    masks = ["| Method | Seed | Features |", "|---|---:|---|"]
    for method in METHODS:
        masks.append(
            f"| {method} | {final[method]['seed']} | {', '.join(final[method]['features'])} |"
        )
    cost_table = [
        "| Search | Calls | Classifier fits | Seconds | Three-seed Jaccard pairs |",
        "|---|---:|---:|---:|---|",
    ]
    for method in SEARCHES:
        runs = [r for r in development["searches"] if r["method"] == method]
        cost_table.append(
            f"| {method} | {sum(r['calls'] for r in runs)} | {sum(r['fits'] for r in runs)} | {sum(r['seconds'] for r in runs):.3f} | "
            + ", ".join(
                f"{p['seeds']}: {p['jaccard']:.3f}" for p in development["stability"][method]
            )
            + " |"
        )
    weighted = scores["all_features_weighted"]["whole"]
    ga = scores["binary_ga"]["whole"]
    observed = f"Binary GA retained {ga['features']}/20 metrics and achieved target MCC {ga['mcc']:.4f}, compared with {weighted['mcc']:.4f} for weighted all-feature LR."
    results = f"""# Measured Camel cross-version results

Research question: can binary feature selection reduce software metrics while preserving cross-version defect detection? {observed}

All eight pre-registered configurations are reported. These are descriptive measurements from one historical release pair; no statistical significance or state-of-the-art claim is made. The final candidate was **{candidate}**, selected exclusively by highest mean outer-development MCC ({development["mean_outer_mcc"][candidate]:.4f}); target scores did not change it. Each optimizer's final mask was chosen independently by pre-registered minimum full-development inner fitness, with feature-count/seed/bitmask tie breaks.

## Whole target: 965 classes, 188 positive

{table_text}

![Verified target comparison](../results/study/figures/target_comparison.png)

## Development evidence, separate from target scores

{chr(10).join(outer_table)}

Three folds are correlated and do not establish significance. Outer fits use masks selected only inside their training part; full-development seed selection is a separate inner-CV procedure. Inner optimized fitness is not an unbiased performance estimate. Source duplicate-vector groups remain disjoint in every inner and outer fold. See `development.json` for every fold index and probability vector.

{chr(10).join(cost_table)}

There were {audited["objective_calls"]} objective calls, {audited["objective_fits"]} actual inner classifier fits, {audited["outer_fits"]} outer classifier fits, and exactly 8 target-stage classifier fits. Empty and repeated masks count; empty masks require no fit. Development took {development["seconds"]:.3f} wall seconds; final fit time was {sum(t["fit_seconds"] for t in timing.values()):.3f} seconds and prediction time {sum(t["predict_seconds"] for t in timing.values()):.3f} seconds. Timing is machine-dependent and objective cost is distinct from final fit cost.

![Fitness](../results/study/figures/fitness_traces.png)
![Search overhead](../results/study/figures/search_cost.png)

## Frozen features

{chr(10).join(masks)}

![Masks](../results/study/figures/frozen_masks.png)

## Seen/unseen-class diagnostics

{chr(10).join(subgroup)}

Subgroups use the same saved probabilities with no additional fits or predictions. The 857 seen and 108 unseen class IDs are temporal-overlap diagnostics. Missing either true class makes F1, recall, precision and AP undefined; MCC is also undefined for any zero denominator. Confusion counts and predicted-positive fractions remain available. Inner fitness maps undefined zero-denominator MCC to 0, explicitly, following sklearn's convention. AP is Average Precision, not trapezoidal PR-AUC.

![Precision recall](../results/study/figures/precision_recall.png)
![Confusion counts](../results/study/figures/confusion_matrices.png)

## Interpretation and limitations

Compare every row with weighted LR, the filter and Random Search before attributing value to GA/GWO. Feature reduction incurs thousands of training fits; favorable target differences alone cannot establish optimizer superiority. The three-seed mask overlaps quantify instability under this fixed, small budget. Negative comparisons remain in the tables.

There are 872 source rows (145 positive) and 965 target rows (188 positive), a prevalence shift from {145 / 872:.2%} to {188 / 965:.2%}. Source data contain 31 duplicate-vector groups and 70 excess rows; target data contain 38 groups and 84 excess rows, including 3 conflicting-label groups. There are 296 shared vectors and 857 shared class IDs. Grouping prevents source CV duplicate-vector leakage, but cannot make the releases independent. Historical defect counts may reflect inconsistent reporting and are not current production ground truth. Metrics share the same predictions and are correlated. Three folds/seeds and population 12 with 90 calls are chosen engineering budgets; conclusions do not generalize to other budgets, learners, projects or future releases. Unseen-class support, especially missing positive support when present, limits conclusions about new classes. No row was removed or relabeled.

## Evidence

Freeze commit: `{audited["freeze_sha"]}`. [Raw artifacts](../results/study/) include data identities, anonymous development fold evidence, every evaluated mask/fitness, chosen masks, environment, freeze, STARTED/COMPLETED chronology, target probabilities and fit times. [Protocol](EXPERIMENT_PROTOCOL.md), [provenance](DATASET_PROVENANCE.md), and [verification](VERIFICATION.md) separate fixed settings from these measured findings. Reports and plots are generated without fitting models.
"""
    (root / "docs/RESULTS.md").write_text(results)
    readme = f"""# Software Defect Prediction

Can binary GA/GWO feature selection reduce software metrics while preserving defect prediction across versions? **{observed}**

![Measured results for all eight frozen configurations](results/study/figures/target_comparison.png)

This completed historical Camel 1.4 → 1.6 experiment measures feature reduction, class imbalance interventions and transfer under a fixed compute budget. The development-selected candidate is **{candidate}**; the eight target scores are descriptive comparisons. This is a portfolio research study, with no claim of a new state-of-the-art method.

## Results and methods

{table_text}

All models use Logistic Regression (`liblinear`, C=1, max_iter=2000, seed=0), threshold 0.5 and training-fold StandardScaler. Baselines include balanced class weights, SMOTE (seed=2029, k=5), and fold-local SelectKBest (ANOVA, k=10). Binary GA, GWO-S, GWO-V and binary Random Search use the real [metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab) package at `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`, with population 12 and 90 objective calls each. The objective combines MCC, AP and feature count; no fitness cache is used.

Camel 1.4 alone supplies three-fold nested StratifiedGroupKFold development (outer seed=2027, inner seed=2028), grouped by identical full metric vectors. Full-development searches use seeds 0, 1, 2; final masks and all settings were committed and pushed before one eight-model target pass. There were **{audited["objective_calls"]} objective calls, {audited["objective_fits"]} inner fits, 24 outer fits, and 8 final fits**. [Full results](docs/RESULTS.md) include all subgroups, masks, timing, stability and negative comparisons.

## Install and verify

Python 3.10, 3.12 and 3.13 are supported by the CI matrix. Git is needed for the pinned optimizer dependency.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
sdp smoke
pytest --cov=software_defect_prediction --cov-report=term-missing
sdp audit-results
```

Smoke uses synthetic data and runs offline. CI never downloads real data or reruns scientific training. See [actual verification evidence](docs/VERIFICATION.md) for local checks, coverage, CI URLs and commit identities.

## Reproduce the study

The checked-in [study artifacts](results/study/) are immutable. Use a fresh output directory for reproduction and follow the [full command sequence](docs/VERIFICATION.md). Downloaded CSVs stay in ignored `data/raw/`. Development reads Camel 1.4 only; `sdp final` checks committed/pushed freeze identities, environment, settings and byte hashes, and creates an exclusive STARTED marker. It refuses repeat scoring in the same output, including after an interrupted pass. [Scientific protocol](docs/EXPERIMENT_PROTOCOL.md) and [specification](docs/PROJECT_SPEC.md) predate training.

## Limits and attribution

This is within-project historical transfer: 857 class IDs and 296 distinct metric vectors overlap across releases. Duplicate vectors occur in both versions; three target duplicate-vector groups have conflicting labels. Source/target defect prevalence differs; metrics are correlated and the three-fold/three-seed budget cannot establish significance or cross-project robustness. See [limitations and seen/unseen support](docs/RESULTS.md#interpretation-and-limitations). Target results never selected the final candidate.

Data: [awsm-research/DefectData](https://github.com/awsm-research/DefectData) revision `7d9a07306ef651fb909344be359dc3523f1db8c5`, Tera-PROMISE CK corpus; Jureczko and Madeyski (2010), DOI [10.1145/1868328.1868342](https://doi.org/10.1145/1868328.1868342). [Provenance and actual SHA-256 hashes](docs/DATASET_PROVENANCE.md) and [machine-readable manifest](results/study/dataset_manifest.json). Raw-data rights remain separate from this code's [MIT license](LICENSE); no raw CSVs are redistributed.

Project 3 in [Ehsan Rezaei's portfolio](https://github.com/Ehsan-Rezaeii), reusing [Project 1: metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab). Source attribution is preserved; optimizers are imported without vendoring.
"""
    (root / "README.md").write_text(readme)
    write_json(output / "audit_report.json", audited)
    write_json(
        output / "artifact_manifest.json",
        {
            str(p.relative_to(output)): sha256(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifact_manifest.json"
        },
    )
    return audited
