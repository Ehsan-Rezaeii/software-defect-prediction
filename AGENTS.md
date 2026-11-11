# AGENTS.md — Software Defect Prediction

## Mission

Implement a complete, maintainable, reproducible research study of within-project cross-version software defect prediction. This is **Project 3** in a five-repository portfolio. Its unique focus is **software quality metrics, class imbalance, feature selection and version-to-version generalization**.

Treat `docs/PROJECT_SPEC.md` and `docs/EXPERIMENT_PROTOCOL.md` as the approved scientific contract. Follow `CODEX_TASK.md` for the complete execution task. All coding, commits, README, figures, and reports must be in professional English.

## Source and dependency rules

- Use exactly the pinned Tera-PROMISE/DefectData **Apache Camel 1.4 (source/development) and Camel 1.6 (later-version target)** files described in `docs/DATASET_PROVENANCE.md`. Do not substitute Ant, NASA, other Camel versions or datasets. Do not invent hashes; compute and record the real SHA-256 of downloaded bytes and check the known Git blob IDs.
- Do not redistribute raw CSV files in the repository; use download scripts, immutable source URLs, manifest checksums and verifiable acquisition procedures. Keep `data/raw/` Git-ignored.
- Depend on the real, installable `metaheuristics-lab` package from commit `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`. Call its `BinaryGA`, `BinaryGWO` and binary `RandomSearch` APIs; do not copy, rewrite, fork, or simulate the optimizer implementations.
- Use simple NumPy/pandas/scikit-learn/imbalanced-learn architecture. No deep learning, AutoML, external paid services, or production API.

## Experimental integrity — mandatory

- The **source release Camel 1.4 is the only dataset for model selection, hyperparameter choices, feature selection, CV, and diagnostics used to select models**. The later release Camel 1.6 may be checked for schema, integrity, counts, identifiers, and overlap in the preregistered audit, but it must not be used to fit, tune, compare candidate models, choose thresholds, or adapt protocol settings before the frozen final stage.
- Freeze all decisions and training artifacts **in a committed and pushed Git commit before accessing target labels for predictive evaluation**. Save the freeze commit and hash in final artifacts. Only then run one final target scoring stage. Do not refreeze after observing target performance or select a winner from target scores.
- For nested CV, group **identical full 20-feature vectors** together in each relevant development fold (prevents exact-vector leakage); use stratification where feasible. Fit scaling, feature filtering and SMOTE only on permitted training-fold data. No global standardization or feature selection.
- Every nonempty optimizer fitness call must fit the prescribed inner-CV classifier from scratch and be counted. Count initialization, repeated masks and empty masks as objective calls. All four searches use equal call budgets, seeds and population size.
- Keep inner optimized fitness, outer development CV, full-development seed selection and final cross-version metrics separate. Do not use the target release to choose a method even if target metrics look excellent or poor.
- Use the exact fixed baselines, costs, search settings, metrics and plots in the protocol. Allow negative conclusions; do not claim significance from 3 folds or 3 seeds.
- No fabricated scores, outputs, citations, test passes, timing, accuracy, or improvement. Reporting must be generated from real artifacts. If the full study cannot complete, state the exact blocker and produce a partial PR rather than inventing completion.

## Engineering and verification

- Python 3.10+; typed, readable `src/` package with clear modules for `data`, `evaluation`, `selection`, `study`, `audit` and a small CLI. Keep experiments orchestration separate from the installable scientific library.
- Pin the direct research dependencies in `pyproject.toml` after compatibility is validated and preserve a machine-readable run environment. `metaheuristics-lab` must remain a direct VCS dependency at the immutable commit.
- Add deterministic pytest tests for data contracts; positive-class semantics; exact objective budgets; optimizer construction; correct metric formulas; fold-group disjointness; leakage detection (including deliberate validation-label perturbation); missing/invalid data; freeze/one-time target guard; and artifact-only metric reconstruction.
- Run Ruff check and format, pytest, synthetic smoke, saved-artifact audit, source/wheel builds and a clean installed-wheel import. CI matrix: Python 3.10, 3.12 and 3.13, **without downloading real data or rerunning the full study**.
- Ensure `ruff format --check .` passes **including any embedded Python code blocks in Markdown files**. Do not let a polished README break CI.
- Any experiment output directory is immutable after final audit; use a new directory for reproduction. Never remove evidence simply to bypass a failed guard.

## Delivery rules

- Commit complete primary study artifacts, human-readable results, plots, provenance and honest limitations. Report all negative findings and method cost. Only claim actual independently checked outcomes.
- `README.md` must lead with the research question, real observed finding and one meaningful plot **after the study runs**; installation, design, reproducibility, and limitations follow. The initial README is explicitly a kickoff placeholder, not a fabricated completed study.
- Produce one open, **unmerged** PR against `main` with complete verification evidence, study/freeze commit SHAs, raw-data checksums, exact fit and objective counts, and any known exceptions.
- No GitHub merges, tags, releases or deployments. No changes to other portfolio repositories. Never write a CV bullet from a planned result.
