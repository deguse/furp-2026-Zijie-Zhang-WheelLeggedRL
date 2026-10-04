"""Verify and rebuild the frozen closeout numbers using only the standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    manifest = json.loads((ROOT / "research/evidence_manifest.json").read_text(encoding="utf-8"))
    bundle = ROOT / "research" / manifest["bundle"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["sha256"]:
        raise SystemExit("Evidence bundle SHA-256 mismatch")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        listed = [entry["path"] for entry in manifest["files"]]
        if len(names) != len(set(names)) or len(listed) != len(set(listed)):
            raise SystemExit("Duplicate evidence path")
        if set(names) != set(listed):
            raise SystemExit("Evidence bundle members differ from the manifest")
        for item in archive.infolist():
            target = (output / item.filename).resolve()
            if not target.is_relative_to(output):
                raise SystemExit("Unsafe archive path")
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise SystemExit(f"Evidence file mismatch: {entry['path']}")
        output.mkdir(parents=True)
        archive.extractall(output)
    metric_paths = ("metadata/verified_metrics.json", "metadata/recovery.csv", "metadata/roll_boundary.csv")
    expected = {path: (output / path).read_bytes() for path in metric_paths}
    subprocess.run([sys.executable, "-B", str(output / "scripts/extract_metrics.py")], cwd=output, check=True)
    for path, data in expected.items():
        if (output / path).read_bytes() != data:
            raise SystemExit(f"Recomputed numbers differ from the frozen evidence: {path}")
    print(f"PASS: {len(manifest['files'])} evidence files verified; metrics byte-identical. Output: {output}")
    print("No training, physical simulation or hardware operation was performed.")

if __name__ == "__main__":
    main()
