"""
Optional curriculum launcher for V4 Plus.

This script simply calls train_v4_plus.py multiple times with increasing difficulty.
It is intentionally simple: each stage starts a fresh run. For stricter continuation
training across stages, you can load the previous model and VecNormalize statistics,
but fresh staged runs are easier to debug.
"""

import subprocess
import sys


STAGES = [
    ["--grid-size", "5", "--max-steps", "300", "--start-mode", "easy", "--timesteps", "500000"],
    ["--grid-size", "7", "--max-steps", "500", "--start-mode", "easy", "--timesteps", "1000000"],
    ["--grid-size", "7", "--max-steps", "600", "--start-mode", "medium", "--timesteps", "1500000"],
    ["--grid-size", "7", "--max-steps", "600", "--start-mode", "medium", "--random-maze", "--timesteps", "2000000"],
]


def main():
    for idx, args in enumerate(STAGES, start=1):
        print(f"\n========== Curriculum stage {idx}/{len(STAGES)} ==========")
        cmd = [sys.executable, "train_v4_plus.py"] + args
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
