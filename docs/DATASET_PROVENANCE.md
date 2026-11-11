# Dataset acquisition and verified pre-study metadata

## Locked source

- Collection: [awsm-research/DefectData](https://github.com/awsm-research/DefectData), Tera-PROMISE CK metrics.
- Source Git commit: `7d9a07306ef651fb909344be359dc3523f1db8c5`.
- **Development:** [`inst/extdata/terapromise/ck/camel-1.4.csv`](https://github.com/awsm-research/DefectData/blob/7d9a07306ef651fb909344be359dc3523f1db8c5/inst/extdata/terapromise/ck/camel-1.4.csv)
- **Later-version target:** [`inst/extdata/terapromise/ck/camel-1.6.csv`](https://github.com/awsm-research/DefectData/blob/7d9a07306ef651fb909344be359dc3523f1db8c5/inst/extdata/terapromise/ck/camel-1.6.csv)

The following facts were verified from the pinned GitHub file contents **before any model fitting or predictive scoring**. The numbers are data/schema checks, not ML results:

| File | Git blob SHA-1 | Bytes | Rows | `bug > 0` | `bug == 0` |
|---|---|---:|---:|---:|---:|
| `camel-1.4.csv` | `5dd841b2533d98dcc7674be4876884c2fd01c12e` | 110,163 | 872 | 145 | 727 |
| `camel-1.6.csv` | `2a83022e493aad7061cf8406db69a1de1e466b2d` | 122,104 | 965 | 188 | 777 |

The CSVs contain 24 columns in this order: `name, version, name, wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc, bug`. **The column name `name` appears twice:** the first occurrence is project name and the third column is the class identifier. Do not naively rely on duplicate header names; validate column positions explicitly. Exactly 20 numeric predictor columns lie between the identifiers and `bug`.

### Pre-study overlap diagnostics

- Camel 1.4 has 872 distinct class IDs; Camel 1.6 has 965.
- There are **857 class IDs shared** across versions. The later target has 108 class IDs absent from the development release.
- Exact 20-feature duplicate-vector groups: **31** within Camel 1.4 (70 excess rows, no conflicting binary labels) and **38** within Camel 1.6 (84 excess rows, including **3 groups with conflicting binary labels**).
- **296 distinct exact feature vectors** occur in both releases.

These observations motivate development-only group-aware CV and honest reporting of within-project temporal overlap. Do **not** delete, deduplicate, train on, or pick a model from target rows on the basis of these diagnostics. Duplicates with conflicting labels in the target are a limitation, not permission to overwrite labels.

## Acquisition contract

1. Download exactly the two immutable `raw.githubusercontent.com` URLs at the pinned revision. Read bytes without text normalization.
2. Compute SHA-256 over each raw file **at acquisition time** and save the actual hashes to `results/study/dataset_manifest.json`. The Git blob identifiers/byte sizes/column counts above are independent fixed checks; verify Git blob SHA-1 when implemented or clearly state which provenance checks were performed.
3. Assert expected row counts, exact positional header, 20 named finite numeric metrics, integer nonnegative bug counts, and expected positive/negative counts. Fail loudly on mismatches.
4. Treat `bug > 0` as positive class `1`. Preserve the class identifier solely as metadata for subgroup reporting; exclude identifier, project, version and bug count from features.
5. Ignore both raw CSV files in Git. The corpus provenance and redistribution permissions are separate from this repository's MIT license; no unverified right to redistribute raw data is claimed.

## Attribution

Collection: `awsm-research/DefectData`, pinned source above. Dataset lineage: Jureczko and Madeyski, *Towards Identifying Software Project Clusters with Regard to Defect Prediction*, PROMISE 2010, DOI [10.1145/1868328.1868342](https://doi.org/10.1145/1868328.1868342). The [DefectData collection](https://github.com/awsm-research/DefectData) publishes descriptive summaries. Some class identifiers overlap between historical releases; do not characterize this as independent-project transfer.

## Acquisition measurements from this implementation

Both immutable URLs were downloaded as raw bytes. Git blob SHA-1, byte sizes, positional headers, all counts, and all overlap/duplicate/conflict checks above passed before scientific training.

| File | Actual SHA-256 |
|---|---|
| `camel-1.4.csv` | `7ad753be3f225aed336b22a0a605b75833341fb131a07413779f0d0889f043e7` |
| `camel-1.6.csv` | `358ef9bb9a0ddf091100b7dec838b92821bb13ae967670c6bd47f8da6dff5138` |

The [machine-readable manifest](../results/study/dataset_manifest.json) preserves the URLs and all measured checks. The collection's pinned [DESCRIPTION](https://github.com/awsm-research/DefectData/blob/7d9a07306ef651fb909344be359dc3523f1db8c5/DESCRIPTION) declares `License: MIT`; a root `LICENSE` file was not found at this revision. That package declaration does not establish blanket rights for every underlying historical corpus. Raw CSV redistribution is avoided. Project 1's optimizer package is MIT licensed; it is installed as an attributed direct dependency and its code is not vendored.
