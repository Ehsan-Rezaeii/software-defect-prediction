# Software Defect Prediction

[![CI](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/workflows/ci.yml)

**A reproducible study of software defect prediction across Apache Camel releases, combining class-imbalance-aware baselines with binary metaheuristic feature selection.** The project develops models on **Camel 1.4**, evaluates their transfer to **Camel 1.6**, and uses an independently auditable, frozen experimental workflow.

**Research question:** Can Binary Genetic Algorithm (GA) and Binary Grey Wolf Optimizer (GWO) select fewer software metrics without sacrificing defect prediction relative to simple baselines and an equally budgeted Random Search?

**Main finding:** On the later release, **binary Random Search selected 8 of 20 metrics** and achieved the highest observed **MCC (0.1951), F1 (0.3789), and Average Precision (0.3513)** among eight configurations. Binary GA and both GWO variants did not exceed this baseline. The model selected *before* target evaluation—**SMOTE + Logistic Regression**—had the highest development mean MCC (0.2930) but scored **0.0975** on Camel 1.6. The unfavorable result was retained rather than selecting a different model after inspecting the target.

> **Scope:** A historical, within-project research experiment—not a production defect detector, an independent-project validation, or evidence of statistical superiority.

## Study at a glance

| Property | Recorded design or observation |
|---|---|
| **Development release** | Apache Camel **1.4**: 872 classes, 145 defect-prone (16.63%) |
| **Later-version target** | Apache Camel **1.6**: 965 classes, 188 defect-prone (19.48%) |
| **Predictors and label** | 20 numeric software metrics; `bug > 0` defines the positive class |
| **Comparators** | 3 all-feature LR baselines, SelectKBest, Binary GA, GWO-S, GWO-V, Random Search |
| **Evaluation** | 3-fold nested, duplicate-vector-grouped development CV; frozen target scoring |
| **Search fairness** | Population 12 and exactly 90 objective calls per search; three full-development seeds |
| **Verification** | 36 passing tests, 60% statement coverage; CI on Python 3.10, 3.12, and 3.13 |

The optimizers are **installed from [metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab/tree/6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e) at a fixed Git commit**, rather than copied into this repository.

## Measured cross-version results

The following are **measured scores on the entire 965-class Camel 1.6 release**, not inner-search fitness or development-CV means. All models use a fixed classification threshold of 0.5. **MCC** (Matthews correlation coefficient) is the primary metric; **AP** means Average Precision, not trapezoidal PR-AUC. Higher values are better.

| Method | Features | MCC | F1 | Defect recall | AP | Missed defects (FN) |
|---|---:|---:|---:|---:|---:|---:|
| All features, standard LR | 20 | 0.1890 | 0.2060 | 0.1277 | 0.3243 | 164 |
| All features, weighted LR | 20 | 0.1151 | 0.3260 | 0.4362 | 0.3187 | 106 |
| All features, SMOTE + LR | 20 | 0.0975 | 0.3126 | 0.4149 | 0.3118 | 110 |
| SelectKBest (k=10), weighted LR | 10 | 0.1398 | 0.3433 | 0.4574 | 0.2847 | 102 |
| Binary GA, weighted LR | 9 | 0.1627 | 0.3556 | 0.4521 | 0.3179 | 103 |
| Binary GWO-S, weighted LR | 10 | 0.1262 | 0.3262 | 0.4043 | 0.2914 | 112 |
| Binary GWO-V, weighted LR | 12 | 0.1388 | 0.3402 | 0.4415 | 0.2992 | 105 |
| **Binary Random Search, weighted LR** | **8** | **0.1951** | **0.3789** | **0.4787** | **0.3513** | **98** |

![Measured MCC and Average Precision on Apache Camel 1.6](results/study/figures/target_comparison.png)

**Reading the comparison:** Random Search had the strongest observed whole-release MCC/F1/AP and used fewer metrics than GA and GWO. Unweighted all-feature LR came close on MCC (0.1890) but detected only **24 of 188** defective classes. Class weighting and SMOTE increased recall, yet neither was consistently better across MCC, precision, and AP. This is why the study reports several metrics and full confusion counts instead of claiming a single universally best classifier.

The development-based candidate remained **SMOTE + LR** because its outer-development mean MCC was **0.2930 ± 0.0431** (sample SD). Its weaker target MCC of 0.0975 illustrates the difficulty of transferring a development choice to a later software release. The target results were **not** used to switch that choice to Random Search.

See [complete results](docs/RESULTS.md) for every method's precision, TN/FP/FN/TP, outer-fold means and variability, selected masks, seed stability, and runtime costs. The [precision–recall curves](results/study/figures/precision_recall.png) and [frozen feature masks](results/study/figures/frozen_masks.png) are also committed.

### Seen versus unseen classes

The versions share **857 class IDs**; Camel 1.6 also contains **108 class IDs absent from Camel 1.4**, of which **17 are defect-prone**. This is temporal transfer **within one project**, not an evaluation on independent software projects.

On those 17 positive unseen-class cases, Binary GA, both GWO variants, Random Search, and the development-selected SMOTE model each detected **3** (recall **0.1765**). SelectKBest detected **4**; unweighted LR detected **none**. These small subgroup counts and weak recall limit any claim about practical usefulness for new classes. The seen/unseen subsets also differ in prevalence, so the gap cannot be attributed solely to whether a class ID was previously observed.

[Full seen/unseen breakdown](docs/RESULTS.md#seenunseen-class-diagnostics) · [Confusion-matrix plots](results/study/figures/confusion_matrices.png)

## Experimental design and safeguards

1. **Data provenance.** Load exactly the pinned Tera-PROMISE CK Camel 1.4 and 1.6 CSVs; validate raw-byte hashes, duplicate headers, schema, numeric predictors, class IDs, and labels. Raw CSVs are not committed. IDs, release metadata, and raw `bug` counts are excluded from model inputs.
2. **Development only on Camel 1.4.** Use 3-fold outer and inner `StratifiedGroupKFold` with seeds `2027` and `2028`. Rows with identical unscaled 20-feature vectors are grouped so no such vector occurs on both sides of a development fold. Training-only scaling, ANOVA filtering, and SMOTE are implemented inside model pipelines.
3. **Fixed classifier and baselines.** All eight configurations use Logistic Regression (`solver="liblinear"`, `C=1`, `max_iter=2000`, `random_state=0`, threshold 0.5). Compare standard LR, balanced class weights, training-fold SMOTE (`random_state=2029`, `k_neighbors=5`), and weighted `SelectKBest(f_classif, k=10)` before attributing value to optimization.
4. **Equal search budgets.** The four imported binary optimizers use population **12** and exactly **90 objective evaluations** per search. Minimize `0.60 × (1 − mean MCC) / 2 + 0.40 × (1 − mean AP) + 0.02 × selected / 20`. Repeated masks count and are re-evaluated; no fitness cache is used. An empty mask receives a penalty without fitting a model.
5. **Freeze before target scoring.** For each search, select its final mask from three full-development seeds (`0, 1, 2`) using the predefined fitness and tie-breaks. Select the study's candidate by highest mean outer-development MCC. Commit and push all settings, masks, environment identity, development artifacts, and source hashes **before** the target stage. Fit each of the eight frozen configurations once on all of Camel 1.4, then evaluate on Camel 1.6.

The recorded [freeze commit](https://github.com/Ehsan-Rezaeii/software-defect-prediction/commit/4140a5124bd76a4710c39da5a9053accccdd26f9) precedes target scoring. The saved `STARTED.json` and `COMPLETED.json` markers, hashes, and immutable study directory support independent auditing. CI uses **synthetic smoke tests and saved-artifact auditing**, not repeated training on the historical target.

### Actual compute accounting

| Measured work | Count |
|---|---:|
| Search executions | 24 (4 methods × [3 outer + 3 full-development]) |
| Objective evaluations | **2,160** |
| Inner classifier fits | **6,480** |
| Outer classifier fits | **24** |
| Final target-stage classifier fits | **8** |
| **Total classifier fits** | **6,512** |

There were **107 repeated-mask objective calls** and **zero empty masks** in the saved searches. Equal optimizer-call budgets do **not** imply equal wall-clock time, nor do they equalize search costs against methods that do no feature search. [Fitness traces](results/study/figures/fitness_traces.png) and [cost analysis](results/study/figures/search_cost.png) document that distinction.

For exact split indices, metric definitions, selection rules, and stopping safeguards, see the [experimental protocol](docs/EXPERIMENT_PROTOCOL.md) and [implementation notes](docs/IMPLEMENTATION.md).

## Install and verify

**Requirements:** Python 3.10+, Git, and network access for initial installation of the pinned optimizer dependency. The project's CI explicitly tests Python **3.10, 3.12, and 3.13**; the recorded scientific environment uses Python 3.12.

```bash
git clone https://github.com/Ehsan-Rezaeii/software-defect-prediction.git
cd software-defect-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'

ruff check .
ruff format --check .
python scripts/check_markdown.py
python -m pytest --cov=software_defect_prediction --cov-report=term-missing
sdp smoke
sdp audit-results
python -m build
```

The smoke test is **synthetic and offline**. `sdp audit-results` reconstructs metrics and checks budgets, fold boundaries, freeze chronology, and saved hashes **without fitting a classifier or downloading the dataset**. The last observed main-branch CI succeeded on all three configured Python versions. Its tests reported **36 passed with 60% overall statement coverage**; coverage is not claimed to be comprehensive, especially for the CLI and report renderer.

[GitHub Actions](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/workflows/ci.yml) · [Verification record and exact commands](docs/VERIFICATION.md) · [Coverage by module](docs/COVERAGE.txt)

## Reproduce the research workflow

The committed [scientific artifacts](results/study/) are sealed. **Do not remove their one-time markers, overwrite them, or rerun `sdp report` over that directory**: reporting also generates README/results files. A computational reproduction requires a **fresh output directory and separate Git branch**, with a new committed and pushed freeze before target scoring.

The following is the Linux/macOS sequence from the documented workflow. Run it from a clean checkout on a reproduction branch that you control:

```bash
git switch -c reproduce-camel-study
study_output=results/reproduction-001
sdp download --output "$study_output"
git add "$study_output/dataset_manifest.json"
git commit -m "Record reproduction dataset provenance"
git push -u origin HEAD

sdp develop --output "$study_output"
git add "$study_output"
git commit -m "Record reproduced development evidence"
git push

sdp freeze --output "$study_output"
sha256sum "$study_output/freeze.json" > "$study_output/freeze.sha256"
git add "$study_output/freeze.json" "$study_output/freeze.sha256"
git commit -m "Freeze reproduction before target evaluation"
git push

sdp final --output "$study_output"
sdp audit-results --output "$study_output"
```

The final stage checks the committed/pushed freeze HEAD, environment, data, and protected file hashes. Reproducing the already reported target is **not** a fresh independent validation. See [the recorded reproduction and verification procedure](docs/VERIFICATION.md#fresh-reproduction).

## Evidence and repository guide

| Resource | Contents |
|---|---|
| [Full measured results](docs/RESULTS.md) | Eight-method target results, development estimates, seen/unseen diagnostics, masks, and limitations |
| [Pre-run experimental protocol](docs/EXPERIMENT_PROTOCOL.md) | Fixed research question, budgets, folds, models, and evaluation rules |
| [Dataset provenance](docs/DATASET_PROVENANCE.md) | Exact upstream revision, raw SHA-256 hashes, duplicate and overlap audits |
| [Freeze manifest](results/study/freeze.json) | Selected masks, study settings, identity hashes, and provenance |
| [Saved predictions](results/study/target_predictions.json) | Anonymized row-index scores and labels supporting independent metric checks |
| [Artifact audit](results/study/audit_report.json) | Search evaluation and fit counts plus freeze identity |
| [Verification and CI evidence](docs/VERIFICATION.md) | Executed commands, fold checks, package builds, test results, and limitations of coverage |
| [Implementation](src/software_defect_prediction/) | Modular source for data, evaluation, selection, study stages, audits, and reporting |

## Limitations and interpretation

- **Historical within-project transfer only:** class-ID overlap (857) and shared exact metric vectors (296) prevent interpreting Camel 1.6 as a fully independent population. Group-aware source CV does not eliminate cross-version dependence.
- **Dataset quality:** there are 31 duplicate-vector groups in Camel 1.4 and 38 in Camel 1.6, including **three target groups with conflicting binary labels**. Rows and labels were preserved, not silently removed or corrected.
- **Weak performance on new classes:** just 17 positive unseen-ID cases yield unstable subgroup estimates; GA, GWO, Random Search, and SMOTE each detected only 3 of them.
- **Limited design:** one classifier family, one fixed threshold, three CV folds and search seeds, and small optimization budgets. No hypothesis test or evidence of general optimizer superiority is provided.
- **Model-selection transfer:** the development-selected SMOTE candidate underperformed on the target. Random Search's higher target MCC is a **post-evaluation observation**, not a reason to change the frozen candidate.
- **Engineering scope:** CI establishes successful tests, packaging, and artifact consistency; it does not independently rerun training on real data. Overall statement coverage is 60%, with weaker coverage of CLI and reporting paths.

## Data attribution and portfolio context

The raw metric tables were obtained from [`awsm-research/DefectData`](https://github.com/awsm-research/DefectData) at commit [`7d9a073`](https://github.com/awsm-research/DefectData/tree/7d9a07306ef651fb909344be359dc3523f1db8c5), following the Tera-PROMISE/CK software-defect dataset lineage. Dataset background: Jureczko & Madeyski (2010), *Towards Identifying Software Project Clusters with Regard to Defect Prediction*, [DOI: 10.1145/1868328.1868342](https://doi.org/10.1145/1868328.1868342). The project records exact raw-data hashes but does **not** redistribute the raw CSVs or assume rights from an upstream package license.

This is the **software-quality / cross-version generalization** application in a five-project portfolio. It reuses the [metaheuristics-lab](https://github.com/Ehsan-Rezaeii/metaheuristics-lab) optimizer package as a real pinned dependency. The other applications study [breast cancer feature selection](https://github.com/Ehsan-Rezaeii/breast-cancer-feature-selection), [SQL injection detection](https://github.com/Ehsan-Rezaeii/sqli-detection-lab), and [search-based test generation](https://github.com/Ehsan-Rezaeii/search-based-test-generation). Their metrics and datasets are different, so the numerical outcomes should not be compared directly.

**Code license:** [MIT](LICENSE). Upstream dataset and publication rights remain separate.

