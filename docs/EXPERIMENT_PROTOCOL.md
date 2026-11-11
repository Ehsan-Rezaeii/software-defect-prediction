# Pre-registered experiment protocol — Project 3 (Camel)

**Status: PRE-RUN. No model-performance result has been measured by this starter pack.**

This scientific protocol must be committed before running search, model selection or target prediction. A code change required by a failed preflight test should be committed and documented **before** the full development run; any material scientific change requires an amended protocol and a fresh study, not an adjustment based on target performance.

## Primary question and design

Can binary GA/GWO feature selection reduce the number of software metrics without degrading defect prediction relative to class-imbalance-aware baselines? Evaluate **within-project cross-version transfer**: Camel **1.4** for all development; Camel **1.6** reserved for final evaluation. The available target is a historical, public release, not a blind industrial trial.

### Fixed model comparisons — eight configurations

1. `all_features`: 20 features, standard Logistic Regression.
2. `all_features_weighted`: 20 features, Logistic Regression with `class_weight="balanced"`.
3. `all_features_smote`: 20 features, train-fold-only SMOTE and Logistic Regression (no class weighting).
4. `select_k_best`: `SelectKBest(f_classif, k=10)`, weighted Logistic Regression.
5. `binary_ga`: optimizer-selected mask, weighted Logistic Regression.
6. `binary_gwo_s`: S-transfer Binary GWO selected mask, weighted Logistic Regression.
7. `binary_gwo_v`: V-transfer Binary GWO selected mask, weighted Logistic Regression.
8. `binary_random_search`: optimizer-selected mask, weighted Logistic Regression.

Shared Logistic Regression: `solver="liblinear"`, `C=1.0`, `max_iter=2000`, `random_state=0`, default threshold 0.5. No target threshold tuning. `StandardScaler` is fit only on the training fold. For SMOTE, use `imblearn.pipeline.Pipeline` with scaling -> SMOTE(`random_state=2029`, `k_neighbors=5`) -> LR, provided minority counts are sufficient; otherwise preflight must stop and be repaired before any target scoring. `SelectKBest` must be fit only inside the current training fold. For selected masks, refit a fresh scaler + LR on the selected columns.

### Development CV

- Preserve all 872 Camel 1.4 rows; no target rows in model development.
- Group rows by **identical 20-feature vectors** to avoid putting the same predictor vector on both train and validation sides. Use `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2027)` for outer CV and another `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2028)` constructed **within each outer training fold** for mask fitness.
- Validate disjoint groups, nonempty binary class support, and availability of enough minority rows for SMOTE in every actual fitting fold **before scientific training**. Do not replace group-aware folds with ordinary stratified folds silently. An approved protocol amendment is required if invalid.
- For each outer fold/method, run feature selection on that fold's training part only, fit the final estimator there and measure outer validation MCC, F1, recall, precision, Average Precision and confusion matrix.
- For each binary search, perform three new full-development searches using optimizer seeds `[0, 1, 2]` and the same inner CV strategy. Choose the mask with minimum inner fitness, breaking ties by `(selected_feature_count, seed, bitmask_as_tuple)`; this decision is made **before** Camel 1.6 is scored.
- Baseline and filter configurations are frozen from development in the same way, without any access to target predictive scores.

### Binary optimizer contract

- Direct dependency: `metaheuristics-lab @ git+https://github.com/Ehsan-Rezaeii/metaheuristics-lab.git@6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`.
- Constructors: `BinaryGA(population_size=12)`, `BinaryGWO(population_size=12, transfer="s")`, `BinaryGWO(population_size=12, transfer="v")`, `RandomSearch(population_size=12, domain="binary")`.
- Exactly **90 objective evaluations** per search, including initialization, duplicate masks and empty masks. No caching of fitness. For each nonempty mask evaluation: fit fresh weighted LR pipelines on the **three inner folds** and compute their validation metrics. Empty mask fitness = `2.0`, with zero fits.
- Minimize `0.60 * (1 - mean_inner_MCC) / 2 + 0.40 * (1 - mean_inner_AP) + 0.02 * selected_feature_count / 20`. MCC lies in [-1,1], AP in [0,1]. Compute means over the three inner folds. This explicit feature penalty is an engineering choice, not a measured dollar cost.
- Count optimizer objective calls, actual classifier fits, and wall times separately. Save every evaluated mask, objective fitness, selected count and best-so-far history. Use consistent seed/fold policies; state that identical seeds do not imply identical candidate sequences across optimizers.
- Expected budget: 4 optimizers x (3 outer + 3 full-development searches) x 90 = **2,160** objective calls. Nonempty fitness calls require 3 classifier fits each, but count real fits rather than assuming every mask is nonempty.

## Final Camel 1.6 evaluation

1. Perform source-only data audit and run all development/fold/smoke tests; validate that the code is ready. Do not compute model performance on Camel 1.6 in preflight or CI.
2. Generate a freeze manifest with exact dataset SHA-256s, source code hashes, all model settings, CV/group seeds, selected masks and seeds, dependency commit, train-derived results, SHA-256 of development artifacts and environment versions. Commit and push this freeze **before** target scoring; preserve its Git SHA.
3. A distinct one-time `final` command must verify frozen code, settings and artifacts and a committed Git freeze. It then fits each of the eight frozen configurations **once** on all of Camel 1.4 and obtains the eight probability vectors on Camel 1.6. Derive all subgroup scores from these same vectors; no extra inference or new fits for subgroups.
4. Report whole-target and prespecified `seen_id`/`unseen_id` subgroups (membership by Camel 1.4 class IDs). Expected 857 seen and 108 unseen target classes from pre-study schema audit. Subgroup classification metrics must be null/undefined when classes or denominators are missing; never silently replace undefined AP/recall with a perfect score.
5. Final metrics: **MCC (primary)**, F1, positive-class recall, precision, Average Precision (AP), TN/FP/FN/TP, predicted-positive fraction, class support/prevalence, feature count. Distinguish AP from trapezoidal PR-AUC. Also report development 3-fold means/sample SD, mask overlap (Jaccard across 3 seeds), runtime and fitness traces.
6. Selection and protocol must not change after target metrics are observed. A negative result is acceptable. No claim of statistical significance, causal effect of overlap, production readiness, or general cross-project robustness.

## Artifacts and CI

- Commit `results/study/` with machine-readable outer-fold rows, masks, traces, seed repeats, selected frozen settings, environment, source provenance, target predictions (anonymous row indices only), final metrics, summary tables, plots and SHA-256 manifest. Keep raw source CSVs uncommitted.
- Provide `docs/RESULTS.md` with a concise result-first narrative and actual numbers; `docs/VERIFICATION.md` with commands and observed evidence; `docs/LIMITATIONS.md` if the primary README would become overly technical.
- Record the commit sequence before freeze and final scoring with immutable markers. CI must run unit tests, synthetic smoke, artifact checks and packaging only. It must **not** download Camel data or perform target scoring.
- The final README must present the actual primary result, practical cross-version interpretation and one checked-in plot, with links to full protocol, raw artifacts and limitations.

## Disclosures and failure behavior

The public target data's label prevalence may be checked during integrity auditing. This does **not** authorize model/feature selection based on target labels. Same class IDs and even identical metric vectors occur across Camel 1.4 and 1.6; treat these as temporal overlap and report them. The target has three conflicting-label identical-vector groups: report data-quality implications without editing source rows. If a scientific protocol step is impossible, **stop** before the final target stage; document the blocker rather than silently substitute a different method or produce fictional results.
