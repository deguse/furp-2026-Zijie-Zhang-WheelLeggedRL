#!/usr/bin/env python3
"""Diagnostic distribution capture for the camp flat-rolling trigger check.

The registered preflight measured 55 stair-mode false latches over 96000 flat
rolling events on the training host (STOP_NO_PROMOTION), while the identical
construction on a CPU development machine measures a flat-metric ceiling of
9.7 N against the 18 N threshold and zero latches. The preregistration's STOP
branch requires re-deriving the threshold/window FROM NEW MEASUREMENTS, and
this script produces exactly that measurement on the device that stopped:
the full per-step `|F0*nx|` distribution plus the location (env, step within
episode, root position, metric value) of every latch.

Read-only and checkpoint-free. It reuses the live adapter's own pretraining
FP backend and session construction, so the environment it measures is the
one the formal check ran; the only difference is instrumentation. Output is
diagnostic evidence for re-registration, NOT campaign evidence:
`evidence_eligible` is pinned false in the payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[3]
if str(SRC_PATH) not in sys.path:
  sys.path.insert(0, str(SRC_PATH))

METRIC_HISTOGRAM_EDGES_N = (
  0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 14.0, 18.0, 20.96, 25.0, 30.0, 50.0, 100.0,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--request", type=Path, required=True,
                      help="pretraining_trigger_request.json from the campaign root")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)

  import numpy
  import torch

  from hoppertrex_mjlab.hybrid.runner import repository_git_sha
  from hoppertrex_mjlab.hybrid.stair_trigger import (
    STAIR_TRIGGER_FORCE_N,
    STAIR_TRIGGER_WINDOW,
    stair_trigger_metric,
  )
  from hoppertrex_mjlab.scripts.rsl_rl import stair_camp_live_adapter as adapter

  request = json.loads(args.request.read_text(encoding="utf-8"))
  request = dict(request)
  # The archived request binds the SHA the campaign validated at. This
  # diagnostic must run at current HEAD, so rebind and let the backend's own
  # provenance checks confirm everything else.
  request["git_sha"] = repository_git_sha()
  request["device"] = args.device
  normalized = adapter.validate_pretraining_trigger_request(request)

  dependencies = adapter._load_live_dependencies()
  backend = adapter._PretrainingFpBackend(normalized, dependencies)
  gates = adapter._formal_pretraining_gate_requests()
  gate = gates["velocity_gate_passed"]
  if gate.name != "velocity_gate_passed" or not gate.commands:
    raise RuntimeError("Velocity gate binding drifted.")

  metric_batches: list[numpy.ndarray] = []
  latches: list[dict[str, object]] = []

  with backend._session(
    domain="flat",
    cells=(0.0,),
    num_envs=gate.num_envs,
    pushes=False,
    purpose="diagnostic_camp_flat_rolling_fp_distribution",
  ) as session:
    wrapped = session.tracker
    env = wrapped.unwrapped if hasattr(wrapped, "unwrapped") else wrapped
    while hasattr(env, "unwrapped") and env.unwrapped is not env:
      env = env.unwrapped
    term = env.action_manager.get_term("hybrid_wheel_leg")
    sensor = env.scene.sensors[term.cfg.stair_trigger_sensor_name]
    robot = env.scene["robot"]
    posture = backend._posture_center(session.env_cfg)

    for vx, yaw in gate.commands:
      wrapped.reset()
      previous_mode = term.stair_mode.clone()
      step_in_episode = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
      )
      for step in range(gate.steps):
        backend._force_commands(wrapped, vx=vx, yaw=yaw, posture=posture)
        wrapped.step(backend._policy_actions(wrapped))

        data = sensor.data
        metric = stair_trigger_metric(
          found=data.found,
          force_contact_frame=data.force,
          normal_global=data.normal,
        )
        metric_batches.append(
          metric.detach().to("cpu", dtype=torch.float64).numpy().copy()
        )

        mode = term.stair_mode.clone()
        rising = mode & ~previous_mode
        if bool(rising.any()):
          positions = robot.data.root_link_pos_w.detach().cpu()
          for env_index in torch.nonzero(rising).flatten().tolist():
            latches.append({
              "command_vx": float(vx),
              "global_step": int(step),
              "step_in_episode": int(step_in_episode[env_index].item()),
              "env": int(env_index),
              "metric_now_n": float(metric[env_index].item()),
              "root_xy_m": [
                float(positions[env_index, 0].item()),
                float(positions[env_index, 1].item()),
              ],
            })
        previous_mode = mode
        done = env.reset_buf.bool()
        step_in_episode += 1
        step_in_episode[done] = 0

  values = numpy.concatenate([batch.reshape(-1) for batch in metric_batches])
  edges = list(METRIC_HISTOGRAM_EDGES_N) + [float("inf")]
  histogram, _ = numpy.histogram(values, bins=edges)
  quantile = lambda p: float(numpy.quantile(values, p))  # noqa: E731

  payload = {
    "schema_version": 1,
    "kind": "stair_camp_flat_trigger_fp_distribution_diagnostic",
    "evidence_eligible": False,
    "git_sha": request["git_sha"],
    "device": args.device,
    "threshold_n": float(STAIR_TRIGGER_FORCE_N),
    "window_steps": int(STAIR_TRIGGER_WINDOW),
    "num_envs": int(gate.num_envs),
    "steps_per_command": int(gate.steps),
    "commands": [list(command) for command in gate.commands],
    "samples": int(values.size),
    "metric_max_n": float(values.max()),
    "metric_quantiles_n": {
      "p99": quantile(0.99),
      "p999": quantile(0.999),
      "p9999": quantile(0.9999),
      "p99999": quantile(0.99999),
    },
    "fraction_at_or_over_threshold": float((values >= 18.0).mean()),
    "histogram_upper_edges_n": [edge for edge in edges[1:]],
    "histogram_counts": [int(count) for count in histogram],
    "latch_count": len(latches),
    "latches": latches,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(f"DIAGNOSTIC_COMPLETE latches={len(latches)} "
        f"max={payload['metric_max_n']:.3f}N "
        f"frac_ge_18N={payload['fraction_at_or_over_threshold']:.2e}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
