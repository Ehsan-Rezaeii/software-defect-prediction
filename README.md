# Software Defect Prediction

**A reproducible research study of software defect prediction using cross-version evaluation, class-imbalance-aware baselines, and binary metaheuristic feature selection.**

**Research question:** Can Binary GA or Binary GWO reduce the number of software metrics while preserving defect prediction quality across software versions?

> **Project status: implementation planned.** This starter repository contains a pre-registered experimental specification. No actual model results, figures, test counts, or performance claims are available yet. Replace this status after the complete experiment and verification have run.

## Study design

- **Dataset:** historical Apache Camel 1.4 for development and Camel 1.6 for later-version evaluation, from the version-pinned Tera-PROMISE CK data collection. Each class has 20 static software metrics; the binary target is `bug > 0`.
- **Comparators:** unweighted/weighted Logistic Regression, SMOTE, SelectKBest(k=10), Binary GA, Binary GWO-S/V, and binary Random Search.
- **Evaluation:** group-aware nested CV on the development version; one frozen cross-version target scoring stage; whole-target and seen/unseen-class diagnostics.
- **Metrics:** Matthews Correlation Coefficient (MCC), F1, recall, Average Precision, false positives/negatives, feature count and runtime.
- **Reuse:** feature-selection algorithms come from the real [metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab) distribution pinned to an immutable Git commit, without copying the optimization code.

The protocol is designed to prevent training/validation leakage and avoid tuning against the later-version target. The historical classes overlap substantially across releases; this is a within-project temporal-transfer study, **not independent-project validation**.

## Specification and data provenance

- [Approved scientific specification](docs/PROJECT_SPEC.md)
- [Pre-registered experimental protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Version-pinned source and verified pre-study metadata](docs/DATASET_PROVENANCE.md)

## Setup (once implementation is complete)

Python 3.10+ and Git are expected. After Codex implements the package and adds the real `pyproject.toml`:

```bash
git clone https://github.com/Ehsan-Rezaeii/software-defect-prediction.git
cd software-defect-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

**Note:** The command above is an intended setup procedure, not evidence that the package has already been implemented or validated. The final README will replace this placeholder with exact verified installation, smoke, reproduction, numerical results, figures and limitations.

## Research integrity

No results are anticipated or guaranteed. The source data must be audited, the development protocol executed, masks frozen, and the later-version evaluation run once with committed evidence. Negative comparisons against simple filters or Random Search are valuable results.

**Code license:** MIT, once a `LICENSE` file is added. Upstream dataset attribution and redistribution rights remain separate; raw dataset CSVs are not committed.
