# Verification and reproduction evidence

The primary scientific run is complete. Local verification and the observed green CI matrix are recorded below. Raw files are ignored, and CI uses committed anonymous artifacts without downloading data or rerunning the study.

## Committed sequence

| Stage | Commit |
|---|---|
| Foundation/provenance, no scoring | `47a9596` |
| Fold-local estimators and metrics | `b3f2a8e` |
| Pinned optimizer adapter and independent tests | `aae7bee` |
| Scientific implementation used for development | `5c8a93b` |
| Audited development artifacts and audit-only hardening | `46df0cb0f8e0846c3eaaa375b626308c8612e34e` |
| Committed and pushed freeze, before target prediction | `4140a5124bd76a4710c39da5a9053accccdd26f9` |

`verify_freeze` independently confirmed that the freeze was the remote branch HEAD, all protected code/config/development hashes matched, raw data matched, and the installed environment equaled development. The exclusive STARTED marker precedes all target fitting; COMPLETED records exactly eight fits and SHA-256 hashes of predictions, metrics, timing and STARTED. No target-guided code, masks or protocol changes followed. Post-target changes are reports and verification evidence only.

## Commands actually executed

Commands used the local `.venv/bin/` executables. The shortened executable names below mean those same installed tools; shell redirections only captured logs.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest tests/test_data.py -q
.venv/bin/sdp download
.venv/bin/python -m build
.venv/bin/pytest tests/test_evaluation.py -q
.venv/bin/pytest tests/test_selection.py -q
.venv/bin/pytest -q
.venv/bin/sdp smoke
.venv/bin/pip freeze --exclude-editable > requirements.lock
# Implementation was committed and pushed before scientific training.
.venv/bin/sdp develop
# Independent development audit was run before committing the development artifacts.
# Development artifacts were committed and pushed.
.venv/bin/sdp freeze
sha256sum results/study/freeze.json > results/study/freeze.sha256
# Freeze manifest/digest were committed and pushed at the SHA above.
.venv/bin/sdp final
.venv/bin/sdp audit-results
.venv/bin/sdp report
.venv/bin/ruff check .
.venv/bin/ruff format --check .
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/check_markdown.py
.venv/bin/pytest --cov=software_defect_prediction --cov-report=term-missing --cov-report=xml
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 .venv/bin/sdp smoke
.venv/bin/sdp audit-results
.venv/bin/python -m build
python -m venv /tmp/sdp-wheel
/tmp/sdp-wheel/bin/pip install dist/*.whl
cd /tmp
/tmp/sdp-wheel/bin/python -c 'import software_defect_prediction, metaheuristics_lab; print(software_defect_prediction.__file__, metaheuristics_lab.__file__)'
/tmp/sdp-wheel/bin/sdp smoke
```

The synthetic smoke test also patches socket access and checks for raw-data paths. All eight methods run on generated data only. The artifact test patches `LogisticRegression.fit` to raise, then independently reconstructs target and outer confusion metrics/AP, folds, budgets, masks and hashes. No classifier is fit during audit. Tampered budgets and freeze identities have explicit rejection tests; a synthetic final pass independently counts eight real classifier fits and proves repeat refusal.

`ruff check .` and `ruff format --check .` pass. The explicit Markdown checker found zero Python fences in README/docs; all existing command fences are Bash. No Python fence was excluded from checking. Long report prose literals have the documented E501 exception, not a formatter exemption.

Clean sdist and wheel builds succeeded. The wheel installed in `/tmp/sdp-wheel` outside the checkout, imported both packages from that environment's site-packages, and ran synthetic smoke. No checkout-local optimizer copy or vendored implementation is used. `requirements.lock` and `results/study/environment.json` record the actual Python 3.12 scientific environment.

## Scientific accounting and fold evidence

Every search has population 12 and 90 calls. There are 24 searches: four methods × (three outer searches + three full-source seed repeats). Each search method consumed 540 calls and 1,620 classifier fits. Total: **2,160 calls, 6,480 inner fits, 24 outer fits, eight target fits = 6,512 classifier fits**. There were zero empty masks and 107 repeated-mask calls within searches; repeated masks were evaluated and fitted again. Objective/search runtimes and final training/inference runtimes are saved separately.

All source rows are retained: 872 classes, 145 positive, 727 negative, and 802 distinct full-vector groups. Each actual inner training fold has at least six minority rows; every actual train/validation fold contains both classes. Every group intersection is zero. The table below lists rows / distinct groups / positive rows; the inner indices are local to their outer-training part.

| Boundary | Train rows/groups/positive | Validation rows/groups/positive |
|---|---:|---:|
| Outer 0 | 588/535/92 | 284/267/53 |
| Outer 0, inner 0 | 377/357/66 | 211/178/26 |
| Outer 0, inner 1 | 407/357/64 | 181/178/28 |
| Outer 0, inner 2 | 392/356/54 | 196/179/38 |
| Outer 1 | 585/535/91 | 287/267/54 |
| Outer 1, inner 0 | 394/357/65 | 191/178/26 |
| Outer 1, inner 1 | 397/355/55 | 188/180/36 |
| Outer 1, inner 2 | 379/358/62 | 206/177/29 |
| Outer 2 | 571/534/107 | 301/268/38 |
| Outer 2, inner 0 | 384/357/70 | 187/177/37 |
| Outer 2, inner 1 | 375/356/75 | 196/178/32 |
| Outer 2, inner 2 | 383/355/69 | 188/179/38 |
| Full-source inner 0 | 586/534/99 | 286/268/46 |
| Full-source inner 1 | 578/535/98 | 294/267/47 |
| Full-source inner 2 | 580/535/93 | 292/267/52 |

Development ran in 21.26868373100001 wall seconds. All seeds, source indices, validation probabilities, traces and pairwise Jaccard stability are in [development.json](../results/study/development.json).

## Fresh reproduction

Do not rerun development/final into `results/study/`, remove its markers, or overwrite its artifacts. A fresh reproduction must have its own commits and pushed freeze; these commands deliberately require Git recording between stages. Keep the scientific settings unchanged.

```bash
study_output=results/reproduction-001
sdp download --output "$study_output"
git add "$study_output/dataset_manifest.json"
git commit -m 'Record reproduction provenance'
git push
sdp develop --output "$study_output"
git add "$study_output"
git commit -m 'Record source-only reproduction artifacts'
git push
sdp freeze --output "$study_output"
sha256sum "$study_output/freeze.json" > "$study_output/freeze.sha256"
git add "$study_output/freeze.json" "$study_output/freeze.sha256"
git commit -m 'Freeze reproduction before target scoring'
git push
sdp final --output "$study_output"
sdp audit-results --output "$study_output"
```

The report command was executed once for the primary output. It writes the primary README/results narrative and seals the output; do not rerun it over the sealed study or replace primary findings with a reproduction. The final command requires exact development environment identity, frozen files, current committed freeze HEAD and a matching remote branch HEAD. An interrupted final stage remains STARTED and must not be retried in that output.

## GitHub Actions

The committed workflow has Python 3.10, 3.12 and 3.13 jobs. CI runs lint/format, Markdown fences, pytest with coverage, synthetic smoke, saved-artifact audit, build and isolated wheel import/smoke. CI does not download the dataset or execute `develop`/`final`. The [observed successful PR run](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/runs/43008485197) tested head SHA `6f63f4f9763c9bc8e5b839efa4a24944b8064980`. All steps and all three jobs completed successfully:

| Python | Actual job | Observed conclusion |
|---|---|---|
| 3.10 | [130481826818](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/runs/43008485197/job/130481826818) | success |
| 3.12 | [130481826850](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/runs/43008485197/job/130481826850) | success |
| 3.13 | [130481826394](https://github.com/Ehsan-Rezaeii/software-defect-prediction/actions/runs/43008485197/job/130481826394) | success |

The run includes uploaded coverage XML artifacts for each Python version. This evidence was observed through GitHub's run/job API, not inferred from configuration or local tests. The subsequent documentation-only commit records these observations; the PR's latest-head checks are also verified at handoff. [Open, unmerged PR](https://github.com/Ehsan-Rezaeii/software-defect-prediction/pull/1).

## Observed local completion checks

The final local run passed **36 tests, zero failures and zero skips**, in 8.35 seconds on Python 3.12.14. Overall statement coverage was **60% (419/701 statements)**; the artifact auditor reached 99%, estimator/metric module 90%, and optimizer adapter 96%. The CLI and report generator were executed as commands but are not covered by the pytest measurement. The [complete pytest coverage output](COVERAGE.txt) records the per-module figures and uncovered lines; no coverage threshold is claimed.

The final `ruff check .`, `ruff format --check .`, explicit Markdown check, synthetic offline smoke, sealed saved-artifact audit and clean sdist/wheel builds all passed. The inspected target-comparison figure agrees with the saved eight-method summary table. The freeze/STARTED/COMPLETED machine timestamps remain in their original artifacts; they demonstrate a committed/pushed freeze before the sole predictive target pass.


The final built wheel was additionally installed into a second fresh environment after the README/results were generated. Both imports resolved under `/tmp/sdp-final-wheel/lib/python3.12/site-packages/`; Project 1's installed `direct_url.json` confirmed commit `6b4ef0fa3d72f608146f3dd7eabb84bf39fa896e`. Its smoke run also passed with HTTP(S) proxies pointed to an unreachable local port. The exact additional commands were:

```bash
python -m venv /tmp/sdp-final-wheel
/tmp/sdp-final-wheel/bin/pip install dist/*.whl
cd /tmp
/tmp/sdp-final-wheel/bin/python -c 'import software_defect_prediction, metaheuristics_lab; print(software_defect_prediction.__file__, metaheuristics_lab.__file__)'
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 /tmp/sdp-final-wheel/bin/sdp smoke
```

`sdp audit-data` was also executed successfully after the run, reconfirming both releases' fixed identity/schema/count/duplicate metadata without fitting a model. A Git diff from freeze to final source/config/tests/CI is empty; the scientific implementation and artifacts were preserved.
