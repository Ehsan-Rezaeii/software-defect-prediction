import re
from pathlib import Path

import pytest

from software_defect_prediction.audit import audit_development, audit_results
from software_defect_prediction.data import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/study"


def test_saved_artifact_independent_audit_without_model_fits(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def forbidden(*args, **kwargs):
        raise AssertionError("Artifact audit attempted a model fit")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    if not (OUTPUT / "COMPLETED.json").exists():
        pytest.skip("Primary scientific run not yet completed")
    counts = audit_results(ROOT, OUTPUT)
    assert counts["objective_calls"] == 2160 and counts["target_fits"] == 8
    sealed = read_json(OUTPUT / "artifact_manifest.json")
    assert set(sealed) == {
        str(p.relative_to(OUTPUT))
        for p in OUTPUT.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    for name, digest in sealed.items():
        assert sha256((OUTPUT / name).read_bytes()) == digest, name


def test_artifact_budget_corruption_rejected(tmp_path):
    if not (OUTPUT / "development.json").exists():
        pytest.skip("Primary development not yet completed")
    from software_defect_prediction.data import write_json

    for name in ["development.json", "source_rows.json", "dataset_manifest.json"]:
        (tmp_path / name).write_bytes((OUTPUT / name).read_bytes())
    d = read_json(tmp_path / "development.json")
    d["searches"][0]["calls"] = 89
    write_json(tmp_path / "development.json", d)
    with pytest.raises(AssertionError):
        audit_development(tmp_path)


def test_readme_relative_links_resolve():
    for link in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
        if "://" not in link and not link.startswith("#"):
            assert (ROOT / link.split("#")[0]).exists(), link


def test_raw_csvs_are_ignored():
    from software_defect_prediction.study import git

    assert not any(
        name.endswith(".csv") and "data/raw" in name
        for name in git("ls-files", root=ROOT).splitlines()
    )
    assert "data/raw/" in (ROOT / ".gitignore").read_text()
