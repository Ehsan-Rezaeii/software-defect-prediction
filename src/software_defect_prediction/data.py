"""Positional CSV parsing and immutable source identity checks; no model fitting."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REVISION = "7d9a07306ef651fb909344be359dc3523f1db8c5"
FEATURES = (
    "wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc".split()
)
HEADER = ["name", "version", "name", *FEATURES, "bug"]
EXPECTED = {
    "1.4": ("5dd841b2533d98dcc7674be4876884c2fd01c12e", 110163, 872, 145, 31, 70, 0),
    "1.6": ("2a83022e493aad7061cf8406db69a1de1e466b2d", 122104, 965, 188, 38, 84, 3),
}


@dataclass(frozen=True)
class Release:
    x: np.ndarray
    y: np.ndarray
    ids: tuple[str, ...]


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def parse(raw: bytes, version: str) -> Release:
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    if not rows or rows[0] != HEADER or any(len(row) != 24 for row in rows[1:]):
        raise ValueError("Invalid positional CSV header/row width")
    body = rows[1:]
    if not body or any(row[0] != "camel" or row[1] != version for row in body):
        raise ValueError("Invalid release metadata")
    x = np.asarray([row[3:23] for row in body], dtype=float)
    bugs = np.asarray([row[23] for row in body], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(bugs).all():
        raise ValueError("Nonfinite data")
    if np.any(bugs < 0) or np.any(bugs != np.floor(bugs)):
        raise ValueError("Invalid bug counts")
    ids = tuple(row[2] for row in body)
    if len(set(ids)) != len(ids) or any(not name for name in ids):
        raise ValueError("Invalid class IDs")
    return Release(x, (bugs > 0).astype(int), ids)


def groups(x: np.ndarray) -> np.ndarray:
    """Group equality is numerical equality of the full, unscaled 20-vector."""
    return np.unique(x, axis=0, return_inverse=True)[1]


def metadata(data: Release) -> dict:
    g = groups(data.x)
    counts = np.bincount(g)
    return {
        "rows": len(data.y),
        "positive": int(data.y.sum()),
        "negative": int((data.y == 0).sum()),
        "class_ids": len(set(data.ids)),
        "duplicate_groups": int((counts > 1).sum()),
        "excess_rows": int((counts - 1).sum()),
        "conflict_groups": sum(len(set(data.y[g == k])) > 1 for k in np.flatnonzero(counts > 1)),
    }


def verified(raw: bytes, version: str) -> tuple[Release, dict]:
    blob, size, rows, pos, dup, excess, conflicts = EXPECTED[version]
    actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if len(raw) != size or actual_blob != blob:
        raise ValueError("Raw Git identity mismatch")
    data = parse(raw, version)
    meta = metadata(data)
    if list(meta.values()) != [rows, pos, rows - pos, rows, dup, excess, conflicts]:
        raise ValueError("Pre-study metadata mismatch")
    return data, {"sha256": sha256(raw), "git_blob": actual_blob, "bytes": size, **meta}


def load_source(raw_dir: Path) -> Release:
    """The development entry point has no target path or target argument."""
    return verified((raw_dir / "camel-1.4.csv").read_bytes(), "1.4")[0]


def acquire(raw_dir: Path, output: Path) -> dict:
    manifest = {"revision": REVISION, "files": {}}
    releases = {}
    for version in EXPECTED:
        name = f"camel-{version}.csv"
        url = (
            f"https://raw.githubusercontent.com/awsm-research/DefectData/{REVISION}/"
            f"inst/extdata/terapromise/ck/{name}"
        )
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        data, identity = verified(raw, version)
        releases[version] = data
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Refusing to replace raw bytes")
        path.write_bytes(raw)
        manifest["files"][name] = {"url": url, **identity}
    a, b = releases.values()
    overlap = len(set(a.ids) & set(b.ids))
    vector_overlap = len(set(map(tuple, a.x)) & set(map(tuple, b.x)))
    if (overlap, vector_overlap) != (857, 296):
        raise ValueError("Cross-version overlap mismatch")
    manifest["overlap"] = {"class_ids": overlap, "vectors": vector_overlap, "unseen_ids": 108}
    path = output / "dataset_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Immutable manifest mismatch")
    write_json(path, manifest)
    return manifest
