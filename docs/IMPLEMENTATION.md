# Implementation decisions recorded before scientific training

The approved scientific protocol is unchanged. These details resolve implementation conventions before development or target scoring.

- The outer search seed is the zero-based outer fold index (0, 1, 2), identical across the four searches. Full-development seeds remain 0, 1, 2. Identical seeds need not generate identical candidate sequences across algorithms.
- The overall final candidate is chosen by highest mean outer-development MCC; ties use fewer frozen features then method name. Each search method independently freezes its prescribed full-development seed winner; all eight methods are scored regardless of candidate status.
- Report MCC as undefined (JSON null) for a zero denominator. F1, recall, precision and AP are null if either true class is missing; each metric is also null for its own missing denominator. Confusion counts, support, prevalence and predicted-positive fraction remain defined. In the inner objective only, undefined zero-denominator MCC maps to zero, as in scikit-learn. Every actual inner fold must have both true classes, so its AP is defined.
- Scaling precedes the ANOVA filter inside each fitting pipeline. Full-source filter support is frozen using preprocessing alone; the final pipeline refits that filter on source rows and asserts identical support. This preprocessing fit is not a classifier fit.
- Classifier fit counts increment after successful pipeline fits. Each nonempty objective call requires three fits; empty calls require none. Repeated masks always fit again. Wall times are recorded separately for objective calls, searches, development, and final training/inference.
- The preflight validates all outer, inner-within-outer, and full-source inner folds before any scientific classifier fit. Grouping uses equality of the original full 20-feature vectors. No row is dropped.
- `requirements.lock` records the measured Python 3.12 environment. CI resolves the fixed direct dependencies on Python 3.10/3.12/3.13; platform/Python-specific transitive versions may differ. The scientific final stage requires the exact development environment.
- Immutable output guards preserve STARTED evidence after failures. Reports derive from saved probabilities, with no refitting; the final artifact manifest seals all output files. Reproductions require a new output directory and their own committed/pushed freeze.
- Python Markdown fences are checked explicitly by `scripts/check_markdown.py`, because Ruff itself does not traverse Markdown documents. Long prose literals in the report generator have a file-specific line-length exception; Python formatting and all other lint rules still apply.

## Pre-freeze engineering audit

The full development run completed from code commit `5c8a93b` in 21.269 seconds with 2,160 calls, 6,480 inner fits and 24 outer fits. The independent development audit passed. Before freeze, the artifact auditor gained explicit Git ancestry/manifest-byte verification and a final sealed-output hash check; these audit-only additions do not change training, masks, methods, predictions or the scientific protocol. Actual acquired hashes and the upstream package-license declaration were also documented. No target predictive evaluation had occurred at that point.
