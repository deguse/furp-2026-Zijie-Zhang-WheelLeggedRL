# Reproducing the research release

## Supported runtime

Windows x64, Python 3.11, a suitable NVIDIA GPU/driver for GPU rollouts. The frozen runtime used Torch 2.11.0+cu128, MuJoCo 3.8.1.dev913242127, mujoco-warp 3.9.0.1 and warp-lang 1.14.0. `uv.lock` is authoritative. Do not upgrade physics dependencies when comparing historical evidence.

Clone the two public repositories as siblings. From a new parent directory:

```powershell
git clone https://github.com/deguse/furp-2026-Zijie-Zhang-WheelLeggedRL.git
Set-Location furp-2026-Zijie-Zhang-WheelLeggedRL
git checkout research-closeout-2026-10-03
.\scripts\bootstrap_research.ps1
```

The bootstrap clones the missing sibling `mjlab-main` at commit `43e0f3ea9c92ddbb4de9f3bb1ac772d604e3ebf6`, verifies the vendored MuJoCo wheel and runs `uv sync --frozen --python 3.11`. It refuses to alter an existing dirty or differently pinned MjLab checkout. It does not train, launch a viewer, change curriculum or operate hardware. Git and uv must already be installed. Historical Linux instructions do not imply that this Windows-specific lockfile supports Linux.

The Python sources use a namespace package. When opening a new terminal, set:

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/src/hoppertrex_mjlab"
```

## Offline results and checkpoints

Python 3.11 alone is sufficient to validate the public evidence bundle and regenerate the reported numbers:

```powershell
python -B scripts/reproduce_closeout.py --output ../closeout_verify
```

The destination must not exist. The script verifies the bundle and individual SHA-256 hashes, preserves historical source SHAs, and compares the recomputed metrics byte for byte. Checkpoints are extracted to `evidence/stage5/model_99.pt` and `evidence/camp/model_999.pt`. They belong to different experiments; neither is permission to bypass staged promotion checks. Only load the files after checksum verification.

For figures, install the extracted `metadata/docs_requirements.txt` in a separate documentation environment and use the extracted `scripts/make_figures.py`. The `sources/latex` directory contains the frozen report sources. The original 2026-09-29 archive on the workstation is not modified by this procedure.

## Research versus qualification

The release includes development-only R0c effort/control, motor codec and support-transfer code so that work is not lost. Unit tests validate software contracts, not safe physical deployment. No USB-CAN integration, reliable stair-climbing result or hardware validation is claimed. Historical launchers are revision-specific; inspect their exact `--help` and provenance before any costly run. No new training or physics experiment was performed to publish this release.

## Software checks

The exact non-physics test selection and results are recorded in `validation/`. Physics-runtime tests are explicitly excluded from publication-time checks; they are not silently counted as passes. The historical workspace scripts under `research/legacy_workspace_scripts` and poster-authoring scripts are provenance materials, not supported application entrypoints.

To install the test runner in your selected project environment (without changing the runtime lockfile):

```powershell
uv pip install --python .venv/Scripts/python.exe pytest==9.1.1
.\tests\powershell\test_bootstrap_native.ps1
```

`validation/clean_environment.json` records the isolated-install verification. The complete GitHub download stalled during this audit; an independent local Git clone of the same remote-verified commit was used instead. It did not reuse the original virtual environment.
