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
    expected = (output / "metadata/verified_metrics.json").read_bytes()
    subprocess.run([sys.executable, "-B", str(output / "scripts/extract_metrics.py")], cwd=output, check=True)
    if (output / "metadata/verified_metrics.json").read_bytes() != expected:
        raise SystemExit("Recomputed numbers differ from the frozen report")
    print(f"PASS: {len(manifest['files'])} evidence files verified; metrics byte-identical. Output: {output}")
    print("No training, physical simulation or hardware operation was performed.")

if __name__ == "__main__":
    main()
