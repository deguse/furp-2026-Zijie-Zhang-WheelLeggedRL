"""Offline evidence publication checks; no physics runtime or checkpoints loaded."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "metadata/verified_metrics.json",
    "metadata/recovery.csv",
    "metadata/roll_boundary.csv",
)


@pytest.fixture
def reproducer():
    spec = importlib.util.spec_from_file_location("reproduce_closeout", ROOT / "scripts/reproduce_closeout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(tmp_path, files, *, listed=None, duplicates=()):
    research = tmp_path / "repo/research"
    research.mkdir(parents=True)
    bundle = research / "sample.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        for name in duplicates:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(name, files[name])
    manifest = {
        "bundle": bundle.name,
        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "files": [
            {"path": name, "bytes": len(files[name]), "sha256": hashlib.sha256(files[name]).hexdigest()}
            for name in (files if listed is None else listed)
        ],
    }
    (research / "evidence_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return research.parent, bundle


def run(reproducer, monkeypatch, root, output):
    monkeypatch.setattr(reproducer, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["reproduce_closeout.py", "--output", str(output)])
    reproducer.main()


@pytest.mark.parametrize("case,message", [
    ("extra", "members differ"),
    ("missing", "members differ"),
    ("duplicate_zip", "Duplicate evidence path"),
    ("duplicate_manifest", "Duplicate evidence path"),
    ("traversal", "Unsafe archive path"),
    ("bundle_hash", "bundle SHA-256 mismatch"),
    ("file_hash", "Evidence file mismatch"),
])
def test_rejects_invalid_bundle_before_extraction(tmp_path, monkeypatch, reproducer, case, message):
    name = "../escape.txt" if case == "traversal" else "record.txt"
    files = {name: b"data"}
    listed = [] if case == "extra" else [name, name] if case == "duplicate_manifest" else [name]
    root, bundle = make_bundle(tmp_path, files, listed=listed, duplicates=[name] if case == "duplicate_zip" else [])
    manifest_path = root / "research/evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "missing":
        manifest["files"].append({"path": "absent.txt", "bytes": 0, "sha256": "0" * 64})
    elif case == "bundle_hash":
        manifest["sha256"] = "0" * 64
    elif case == "file_hash":
        manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(SystemExit, match=message):
        run(reproducer, monkeypatch, root, output)
    assert not output.exists()
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("changed", METRICS)
def test_checks_every_recomputed_metric(tmp_path, monkeypatch, reproducer, changed):
    files = {path: b"original\n" for path in METRICS}
    files["scripts/extract_metrics.py"] = (
        f"from pathlib import Path\nPath({changed!r}).write_bytes(b'changed')\n"
    ).encode()
    root, _ = make_bundle(tmp_path, files)
    with pytest.raises(SystemExit, match="Recomputed numbers differ"):
        run(reproducer, monkeypatch, root, tmp_path / "result")


def test_refuses_existing_output_without_modifying_it(tmp_path, monkeypatch, reproducer):
    output = tmp_path / "result"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(SystemExit) as exc:
        run(reproducer, monkeypatch, tmp_path / "no_repo", output)
    assert exc.value.code == 2
    assert sentinel.read_bytes() == b"keep"


def test_public_bundle_rebuilds_without_presentation_materials(tmp_path):
    manifest = json.loads((ROOT / "research/evidence_manifest.json").read_text())
    with zipfile.ZipFile(ROOT / "research" / manifest["bundle"]) as archive:
        names = archive.namelist()
        assert not any(name.startswith(("deliverables/", "figures/", "sources/latex/")) for name in names)
        assert "scripts/make_figures.py" not in names
        assert "metadata/docs_requirements.txt" not in names
        assert "metadata/evidence_matrix.md" in names
        assert archive.read("scripts/extract_metrics.py") == (ROOT / "research/closeout_source/scripts/extract_metrics.py").read_bytes()
        expected = {path: archive.read(path) for path in METRICS}
    output = tmp_path / "reproduced"
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/reproduce_closeout.py"), "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert "metrics byte-identical" in result.stdout
    assert not (output / "figures").exists()
    assert not (output / "deliverables").exists()
    for path, data in expected.items():
        assert (output / path).read_bytes() == data
