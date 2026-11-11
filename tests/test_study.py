import socket
import subprocess

import numpy as np
import pytest
from sklearn.datasets import make_classification

from software_defect_prediction.data import read_json, sha256, write_json
from software_defect_prediction.study import SETTINGS, git, preflight, start_final, verify_freeze


def test_smoke_offline_without_data_access(monkeypatch):
    import builtins

    from software_defect_prediction.smoke import smoke

    original = builtins.open

    def guarded(file, *args, **kwargs):
        assert "camel-1." not in str(file) and "data/raw" not in str(file)
        return original(file, *args, **kwargs)

    def offline(*args, **kwargs):
        raise AssertionError("Smoke attempted network access")

    monkeypatch.setattr(socket, "socket", offline)
    monkeypatch.setattr(builtins, "open", guarded)
    assert set(smoke()) == set(SETTINGS["methods"])


def test_preflight_and_one_time_marker(tmp_path):
    x, y = make_classification(n_samples=150, n_features=20, random_state=1)
    assert len(preflight(x, y)) == 3
    start_final(tmp_path, "abc")
    assert read_json(tmp_path / "STARTED.json")["freeze_sha"] == "abc"
    with pytest.raises(FileExistsError):
        start_final(tmp_path, "abc")
    with pytest.raises(ValueError):
        preflight(x, np.zeros(150, int))


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    root = tmp_path
    git("init", root=root)
    git("config", "user.name", "Test", root=root)
    git("config", "user.email", "test@example.invalid", root=root)
    output = root / "results"
    output.mkdir()
    (root / "code.py").write_text("x = 1\n")
    write_json(output / "environment.json", {})
    git("add", ".", root=root)
    git("commit", "-m", "development", root=root)
    write_json(
        output / "freeze.json",
        {"settings": SETTINGS, "files": {"code.py": sha256((root / "code.py").read_bytes())}},
    )
    git("add", ".", root=root)
    git("commit", "-m", "freeze", root=root)
    monkeypatch.setattr("software_defect_prediction.study.environment", lambda: {})
    return root, output


def test_committed_freeze_and_hash_guards(frozen_repo):
    root, output = frozen_repo
    assert verify_freeze(root, output, check_remote=False) == git("rev-parse", "HEAD", root=root)
    with pytest.raises(subprocess.CalledProcessError):
        verify_freeze(root, output, check_remote=True)
    (root / "code.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="Dirty"):
        verify_freeze(root, output, check_remote=False)


def test_manifest_tamper_and_wrong_head(frozen_repo):
    root, output = frozen_repo
    path = output / "freeze.json"
    text = path.read_text()
    path.write_text(text + "\n")
    with pytest.raises(ValueError, match="manifest modified"):
        verify_freeze(root, output, check_remote=False)
    path.write_text(text)
    (root / "new.txt").write_text("after freeze")
    git("add", ".", root=root)
    git("commit", "-m", "new head", root=root)
    with pytest.raises(ValueError, match="HEAD"):
        verify_freeze(root, output, check_remote=False)


def test_develop_has_no_target_loader():
    import inspect

    from software_defect_prediction.data import load_source
    from software_defect_prediction.study import develop

    assert "1.6" not in inspect.getsource(load_source)
    assert "target" not in inspect.getsource(develop)


def test_environment_pinned_dependency():
    from software_defect_prediction.selection import DEPENDENCY_SHA
    from software_defect_prediction.study import environment

    assert environment()["optimizer_direct_url"]["vcs_info"]["commit_id"] == DEPENDENCY_SHA


def test_frozen_identity_comparison_even_after_commit(frozen_repo):
    root, output = frozen_repo
    manifest = read_json(output / "freeze.json")
    manifest["files"]["code.py"] = "0" * 64
    write_json(output / "freeze.json", manifest)
    git("add", ".", root=root)
    git("commit", "-m", "bad digest", root=root)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_freeze(root, output, check_remote=False)


def test_final_synthetic_pass_fits_exactly_eight_and_rejects_repeat(tmp_path, monkeypatch):
    """Exercise the final command using only synthetic releases in an isolated directory."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from software_defect_prediction.data import Release
    from software_defect_prediction.study import final

    x, y = make_classification(n_samples=150, n_features=20, random_state=5)
    source = Release(x, y, tuple(f"class{i}" for i in range(150)))
    target = Release(
        x[:60], y[:60], tuple([f"class{i}" for i in range(30)] + [f"new{i}" for i in range(30)])
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "study"
    output.mkdir()
    files = {}
    for name in ["camel-1.4.csv", "camel-1.6.csv"]:
        (raw / name).write_bytes(b"SYNTHETIC TEST ONLY")
        files[name] = {"sha256": sha256(b"SYNTHETIC TEST ONLY")}
    masks = {m: {"mask": [1] * 20} for m in SETTINGS["methods"]}
    masks["select_k_best"]["mask"] = (
        SelectKBest(f_classif, k=10)
        .fit(StandardScaler().fit_transform(x), y)
        .get_support()
        .astype(int)
        .tolist()
    )
    write_json(output / "freeze.json", {"dataset": {"files": files}, "final": masks})
    monkeypatch.setattr(
        "software_defect_prediction.study.verify_freeze", lambda *a: "synthetic-freeze"
    )
    monkeypatch.setattr("software_defect_prediction.study.load_source", lambda *a: source)
    monkeypatch.setattr("software_defect_prediction.study.verified", lambda *a: (target, {}))
    fits = []
    original = LogisticRegression.fit

    def counted(self, *args, **kwargs):
        fits.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", counted)
    result = final(raw, tmp_path, output)
    assert len(fits) == 8 and len(result) == 8
    assert read_json(output / "COMPLETED.json")["classifier_fits"] == 8
    with pytest.raises(FileExistsError):
        final(raw, tmp_path, output)
    assert len(fits) == 8
