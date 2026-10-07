"""Offline validation/aggregation of three completed pilot evaluations."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from hoppertrex_mjlab.hybrid.recovery_pilot import (
  TRAIN_SEEDS, aggregate_comparisons, compare, read_json, require_identity,
  verify_sealed, write_json,
)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--campaign-root", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  args = p.parse_args()
  if args.output.exists():
    raise FileExistsError("Analysis output must be new")
  identity = read_json(args.campaign_root / "campaign.json")
  summaries = []
  for seed in TRAIN_SEEDS:
    directory = args.campaign_root / f"evaluate_seed{seed}"
    require_identity(verify_sealed(directory), identity)
    baseline = read_json(directory / "baseline/trials.json")["trials"]
    candidate = read_json(directory / "candidate/trials.json")["trials"]
    summary = read_json(directory / "summary.json")
    if summary["comparison"] != compare(baseline, candidate):
      raise ValueError("Stored comparison differs from independently reduced trial records")
    summaries.append(summary)
  result = aggregate_comparisons(summaries)
  args.output.mkdir(parents=True)
  write_json(args.output / "decision.json", {**identity, **result})
  with (args.output / "seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["training_seed", "scale", "baseline_penalized_s", "candidate_penalized_s", "fractional_improvement", "candidate_hard_failures"])
    for s in summaries:
      for scale, row in s["comparison"].items():
        writer.writerow([s["training_seed"], scale, row["baseline"]["mean_penalized_recovery_s"],
                         row["candidate"]["mean_penalized_recovery_s"], row["fractional_improvement"], row["candidate"]["hard_failures"]])
  print(result)


if __name__ == "__main__":
  main()
