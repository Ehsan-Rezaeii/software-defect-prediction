# Project 3 — Software Defect Prediction (Fresh Camel Study)

## Status and intended scope

**Specification status: approved for implementation; experiment not yet executed.** This document describes planned work, not observed scientific results. A complete study must be implemented, tested and executed against the separately frozen protocol. Do not invent measured metrics or write a final-results README until artifacts exist.

## Research objective

Investigate whether feature selection with binary GA and binary Grey Wolf Optimization yields a useful trade-off between selected software metrics, defect-detection quality and computation under cross-version transfer. Compare against simple, strong baselines (including imbalanced-learning interventions), without assuming metaheuristics are superior. Focus on *software defect prediction*, not the medical or SQLi feature-selection applications.

## Fixed source

- Apache Camel **1.4** source/development and Camel **1.6** later-version target from the 20-metric Tera-PROMISE CK corpus.
- Exact immutable source Git revision, Git blob identities and verified pre-study counts are in [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md).
- Task is binary `defective = (bug > 0)`; defect-positive label = 1. Repeated header `name` must be handled by column position, never by automatic duplicate-name resolution alone.
- All 20 metrics: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc`.
- Data quality: check finite values, counts, duplicate vectors, class identifiers, class conflict groups, overlap, label distributions and any source anomalies. Retain original rows. No class ID, release, name or raw `bug` as a predictive feature.
- Dataset bytes live outside Git. Record true hashes and method of retrieval. Do not claim blanket redistribution rights.

## Methods and comparisons

Exactly eight fixed configurations (the short names and all details appear in the immutable [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)):

- AllFeatures LR, AllFeatures weighted LR, AllFeatures + SMOTE LR.
- Training-fold SelectKBest(k=10) with weighted LR.
- Binary GA, Binary GWO-S, Binary GWO-V, binary Random Search with weighted LR.

Use **Project 1** as an actual package dependency at commit `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`. The four searches share `population_size=12`, exact `90` objective evaluations per search and explicit reproducible seeds. Penalize empty masks. Keep the prescribed MCC/AP/feature-count objective, fixed classifier and SMOTE choices. Avoid automatic model selection on the later release.

## Validation and leakage protections

- Exactly one primary development protocol: 3-fold nested stratified **group-aware** CV on Camel 1.4, grouped by identical 20-feature vectors.
- All feature selection, scaling and resampling occur within training folds. No target release used in development. Assert this with deliberate leakage-injection unit tests, not just prose.
- Three full-development search seeds per optimizer, deterministic predeclared winner selection. Freeze all final masks/configuration before later-release inference.
- A one-time target-scoring gate records freeze hashes and forbids accidental reruns in the same output. Target scores cannot revise method choice.
- Explicit seen/unseen class-ID subgroup analysis without treating the known cross-version overlap as separate independent projects.

## Core engineering deliverables

Package layout suggestion, with no unnecessary framework abstractions:

```
src/software_defect_prediction/
    __init__.py
    data.py
    evaluation.py
    selection.py
    study.py
    audit.py
    cli.py
experiments/
    smoke_test.py
    plot_results.py
    summarize_results.py
tests/
    test_data.py
    test_evaluation.py
    test_selection.py
    test_study.py
    test_artifacts.py
scripts/
    download_data.py
results/study/              # committed measured artifacts (after execution)
docs/
    PROJECT_SPEC.md
    EXPERIMENT_PROTOCOL.md
    DATASET_PROVENANCE.md
    RESULTS.md               # created after verified execution
    VERIFICATION.md          # created from real checks
.github/workflows/ci.yml
pyproject.toml
README.md
AGENTS.md
```

- Working CLI: `sdp download`, `sdp audit-data`, `sdp develop`, `sdp freeze`, `sdp final`, `sdp audit-results`, `sdp report` (and a synthetic-only smoke command or script). `develop` cannot read target predictive labels; the first target predictive evaluation occurs only at `final`.
- Cleanly installable wheel and `sdp` console entry point. Reproducible local run and explicit package versions.
- Committed source/protocol and SHA-256 manifest before scoring target. Saved predictions and posthoc metrics must allow an artifact-only recomputation without raw source data.
- Honest report of comparisons and runtime costs; no test-set cherry picking or inflated performance claims.
- CI on Python 3.10/3.12/3.13: Ruff lint/format, pytest, synthetic smoke, saved-artifact audit, package build, isolated wheel install/import.
- A final open/unmerged PR; make no remote destructive changes and do not modify other projects.

## Acceptance checklist

1. Correct source identity, labels, schemas, counts, hashes and source attribution; raw data excluded from Git.
2. Four real imported optimizer classes; equal objective budgets and actual population sizes independently verified.
3. Zero leakage through nested CV, masks, scaling, SMOTE or target results; grouped fold boundaries validated.
4. Correct MCC/F1/recall/precision/AP, confusion matrices and undefined-subgroup handling.
5. Complete development searches and mask stability, frozen before the one target evaluation stage.
6. Every reported number reconstructible from committed artifact evidence, clear compute budget and negative findings preserved.
7. Tested Python package, clean CI matrix, reproducible setup, real results figures and limitations.
8. README accurately reflects what was **measured**, not merely planned; all Markdown code fences pass lint/format checks.

## References

- Pinned collection: [awsm-research/DefectData](https://github.com/awsm-research/DefectData).
- Jureczko and Madeyski (2010), *Towards Identifying Software Project Clusters with Regard to Defect Prediction*, DOI [10.1145/1868328.1868342](https://doi.org/10.1145/1868328.1868342).
- [scikit-learn documentation](https://scikit-learn.org/stable/modules/cross_validation.html) and [imbalanced-learn documentation](https://imbalanced-learn.org/stable/).
