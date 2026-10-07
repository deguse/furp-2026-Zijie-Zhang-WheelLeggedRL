"""Supervised simulator worker for the C1 recovery pilot. Not a public launcher."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from hoppertrex_mjlab.hybrid.recovery_pilot import (
  ACTION_STD, DEV_SEEDS, INFO_KEY, PROTOCOL, SCALES, TASK_ID, TEST_SEEDS,
  compare, digest, file_sha, identity, read_json, require_approval,
  summarize, trial_id, trial_metrics, validate_checkpoint_record,
  validate_trials, verify_sealed, write_json,
)
from hoppertrex_mjlab.tasks.recovery_c1_pilot import (
  REPO, RecoveryPilotRunner, configuration, make_env_cfg, make_runner_cfg,
)
from hoppertrex_mjlab.scripts.evaluate_hybrid_c1_flat_gate import (
  REGISTERED_CAPS, aggregate_candidate, evaluation_cells,
)
from hoppertrex_mjlab.scripts.rsl_rl.evaluate_hybrid_gate import (
  NON_WHEEL_GROUND_SENSOR_NAME, _collect_scenarios, _force_posture,
  _force_velocity_command, _pitch, non_wheel_ground_contact,
)
from hoppertrex_mjlab.scripts.rsl_rl.hybrid_gate import (
  _checks_payload, evaluate_capability_suite, to_deterministic_json,
  wheel_target_saturation_threshold,
)


def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def open_env(*, device, seed, count, checkpoint=None, training=False, campaign=None, cpu_smoke=False):
  seed_all(seed)
  cfg = make_env_cfg(play=not training)
  cfg.seed = seed
  cfg.scene.num_envs = count
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.num_envs = count
  if training:
    cfg.recovery_pilot_mode = "cpu_smoke" if cpu_smoke else "training"
    cfg.recovery_pilot_campaign = str(campaign)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  agent = make_runner_cfg()
  agent.seed = seed
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  if checkpoint is None:
    def policy(obs):
      return torch.zeros((count, 6), device=device)
  else:
    runner = RecoveryPilotRunner(wrapped, dataclasses.asdict(agent), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)
  return wrapped, policy, agent


def initial_states(env):
  data = env.scene["robot"].data
  return torch.cat((data.root_link_pos_w - env.scene.env_origins,
    data.root_link_quat_w, data.root_link_vel_w, data.joint_pos, data.joint_vel), dim=1).detach().cpu().numpy()


def run_recovery(*, device, split, output, checkpoint=None, scales=SCALES, diagnostic=False):
  if diagnostic and device != "cpu":
    raise ValueError("Diagnostic collector is CPU-only")
  count = 2 if diagnostic else 32
  rows = []
  seeds = DEV_SEEDS if split == "development" else TEST_SEEDS
  if diagnostic:
    seeds = seeds[:1]
  center_cfg = make_env_cfg(play=True).commands["posture"]
  height, pitch = 0.5 * sum(center_cfg.height_range), 0.5 * sum(center_cfg.pitch_range)
  output.mkdir(parents=True, exist_ok=False)
  for scale in scales:
    for seed in seeds:
      wrapped, policy, _ = open_env(device=device, seed=seed, count=count, checkpoint=checkpoint)
      env = wrapped.unwrapped
      try:
        seed_all(seed)
        env.common_step_counter = 0
        wrapped.reset()
        starts = initial_states(env)
        if not np.isfinite(starts).all():
          raise ValueError("Non-finite reset state")
        terminated = np.zeros(count, dtype=bool)
        reset = np.zeros(count, dtype=bool)
        contact = np.zeros(count, dtype=bool)
        active = False
        reset_original = env._reset_idx
        # Observe safety BEFORE the automatic reset clears state/sensors.
        def observe_reset(ids):
          if active and len(ids):
            index = ids.detach().cpu().numpy()
            reset[index] = True
            contact[:] |= non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).detach().cpu().numpy().astype(bool)
          return reset_original(ids)
        env._reset_idx = observe_reset
        active = True
        traces = {k: [] for k in ("errors", "healthy", "actions", "applied_residual", "wheel_targets", "actuator_force", "root_state", "terminated", "contact", "reset")}
        signs = torch.where(torch.arange(count, device=device) < count // 2, 1.0, -1.0)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("hybrid_wheel_leg")
        for step in range(600):
          _force_velocity_command(env, 0.0, 0.0)
          _force_posture(wrapped, height, pitch)
          if step == 300:
            velocity = robot.data.root_link_vel_w.clone()
            velocity[:, 0] += signs * scale * 0.04
            velocity[:, 4] += signs * scale * 0.06
            robot.write_root_link_velocity_to_sim(velocity, env_ids=torch.arange(count, device=device))
            env.sim.forward()
            env.sim.sense()
          with torch.no_grad():
            actions = policy(wrapped.get_observations()).detach()
            _, _, dones, _ = wrapped.step(actions)
          _force_velocity_command(env, 0.0, 0.0)
          _force_posture(wrapped, height, pitch)
          terminated |= env.reset_terminated.detach().cpu().numpy().astype(bool)
          reset |= dones.detach().cpu().numpy().astype(bool)
          contact |= non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).detach().cpu().numpy().astype(bool)
          d = robot.data
          errors = torch.stack((d.root_link_lin_vel_b[:, 0], d.root_link_ang_vel_b[:, 2],
                                d.root_link_pos_w[:, 2] - height, _pitch(d) - pitch), dim=1)
          healthy = (errors.abs() <= torch.tensor(PROTOCOL["healthy_bands"], device=device)).all(dim=1)
          arrays = {"errors": errors, "healthy": healthy, "actions": actions,
                    "applied_residual": action_term.applied_residual,
                    "wheel_targets": action_term.wheel_targets,
                    "actuator_force": env.sim.data.actuator_force,
                    "root_state": torch.cat((d.root_link_pos_w, d.root_link_quat_w, d.root_link_vel_w), dim=1)}
          for key, value in arrays.items():
            arr = value.detach().cpu().numpy().copy()
            if not np.isfinite(arr).all():
              raise ValueError(f"Non-finite trace: {key}")
            traces[key].append(arr)
          for key, value in (("terminated", terminated), ("contact", contact), ("reset", reset)):
            traces[key].append(value.copy())
        arrays = {k: np.stack(v) for k, v in traces.items()}
        trace_name = f"scale{scale}_reset{seed}.npz"
        np.savez_compressed(output / trace_name, initial_states=starts, **arrays)
        threshold = wheel_target_saturation_threshold(action_term)
        for i in range(count):
          row = {"pairing_id": trial_id(split, scale, seed, i), "split": split,
                 "scale": scale, "reset_seed": seed, "env_id": i, "sign": 1 if i < count // 2 else -1,
                 "initial_state_sha256": hashlib.sha256(starts[i].tobytes()).hexdigest(),
                 "trace_file": trace_name,
                 **trial_metrics(arrays["healthy"][300:, i].tolist(), terminated=bool(terminated[i]),
                    contact=bool(contact[i]), reset=bool(reset[i])),
                 "wheel_target_saturation_fraction": float((np.abs(arrays["wheel_targets"][:, i]) >= threshold).mean()),
                 "leg_residual_saturation_fraction": float((np.abs(arrays["applied_residual"][:, i, 2:]) >= .99 * .035).mean()),
                 "peak_actuator_force_abs": float(np.abs(arrays["actuator_force"][:, i]).max())}
          rows.append(row)
        print(f"[recovery] {split} scale={scale} reset_seed={seed} complete", flush=True)
      finally:
        wrapped.close()
  if not diagnostic:
    validate_trials(rows, split, tuple(scales))
  else:
    write_json(output / "diagnostic.json", {"evidence_eligible": False, "num_envs": count})
  write_json(output / "trials.json", {"trials": rows})
  write_json(output / "summary.json", summarize(rows))
  return rows


def run_c1_cells(device, checkpoint=None, diagnostic=False):
  """Same registered fifteen commands/caps; evaluate both C and H, not retune."""
  if diagnostic and device != "cpu":
    raise ValueError("Diagnostic C1 collector is CPU-only")
  wrapped, policy, _ = open_env(device=device, seed=211, count=2 if diagnostic else 16, checkpoint=checkpoint)
  env = wrapped.unwrapped
  cells = []
  try:
    commands = evaluation_cells(0.05)[:1] if diagnostic else evaluation_cells(0.05)
    for height, pitch, vx in commands:
      wrapped.reset()
      robot = env.scene["robot"]
      term = env.action_manager.get_term("hybrid_wheel_leg")
      values = {k: [] for k in ("height", "pitch", "rate", "lin", "wheel")}
      terminated = 0
      contacts = []
      previous = None
      for step in range(6 if diagnostic else 300):
        _force_velocity_command(env, vx, 0.0)
        _force_posture(wrapped, height, pitch)
        with torch.no_grad():
          wrapped.step(policy(wrapped.get_observations()).detach())
        _force_velocity_command(env, vx, 0.0)
        _force_posture(wrapped, height, pitch)
        terminated += int(env.reset_terminated.sum().item())
        contacts.append(non_wheel_ground_contact(env, NON_WHEEL_GROUND_SENSOR_NAME).detach().cpu())
        wheel = term.wheel_targets.detach().clone()
        if step >= (2 if diagnostic else 100):
          d = robot.data
          for key, v in (("height", d.root_link_pos_w[:, 2] - height), ("pitch", _pitch(d) - pitch),
                         ("rate", d.root_link_ang_vel_b[:, 1].abs()), ("lin", d.root_link_lin_vel_b[:, 0])):
            values[key].append(v.detach().cpu())
          values["wheel"].append(((wheel - previous).square().sum(dim=1)).cpu())
        previous = wheel
      v = {k: torch.stack(a) for k, a in values.items()}
      cells.append({"target_height": height, "target_pitch": pitch, "vx_command": vx,
        "height_rmse": float(v["height"].square().mean().sqrt()),
        "pitch_rmse": float(v["pitch"].square().mean().sqrt()),
        "pitch_error_abs_p95": float(torch.quantile(v["pitch"].abs(), .95)),
        "pitch_rate_abs_p99": float(torch.quantile(v["rate"], .99)),
        "mean_actual_lin_x": float(v["lin"].mean()),
        "velocity_error_abs": abs(float(v["lin"].mean()) - vx),
        "wheel_target_rate_rms": float(v["wheel"].mean().sqrt()),
        "terminated_events": terminated,
        "non_wheel_contact_rate": float(torch.stack(contacts).float().mean()),
        "safety_window_steps": 300})
      print(f"[C1] completed cell {len(cells)}/15", flush=True)
  finally:
    wrapped.close()
  result = {"cells": cells, "caps": REGISTERED_CAPS, **aggregate_candidate(cells, REGISTERED_CAPS)}
  if diagnostic:
    result.update({"flat_gate_passed": False, "evidence_eligible": False})
  return result


def retention(device, checkpoint=None, diagnostic=False):
  if diagnostic and device != "cpu":
    raise ValueError("Diagnostic retention is CPU-only")
  args = SimpleNamespace(seed=211, device=device, num_envs=2 if diagnostic else 32, steps=64 if diagnostic else 3000,
    warmup_steps=4 if diagnostic else 300, window_steps=8 if diagnostic else 800, episode_length_s=1.0e9,
    ablate_leg_residuals=False, profile="formal", progress_interval=500)
  scenes = _collect_scenarios(suite="integrated", task=TASK_ID, checkpoint=checkpoint, args=args)
  checks = evaluate_capability_suite("integrated", scenes, profile="formal")
  return json.loads(to_deterministic_json({"scenarios": scenes, "checks": _checks_payload(checks),
                                           "passed": not diagnostic and all(c.passed for c in checks), "diagnostic": diagnostic}))


def cpu_smoke(output):
  wrapped, _, agent = open_env(device="cpu", seed=0, count=2, training=True,
                              campaign=output, cpu_smoke=True)
  try:
    runner = RecoveryPilotRunner(wrapped, dataclasses.asdict(agent), log_dir=str(output / "smoke_log"), device="cpu")
    obs = wrapped.get_observations()
    with torch.no_grad():
      mean = runner.get_inference_policy()(obs)
    assert torch.equal(mean, torch.zeros_like(mean))
    std = runner.alg.get_policy().distribution.std_param.detach().cpu().tolist()
    assert np.allclose(std, ACTION_STD, atol=1e-7, rtol=0)
    runner.learn(2)
    result = {"status": "PASS", "num_envs": 2, "updates": 2,
            "initial_deterministic_mean_zero": True, "initial_std": std,
            "formal_training": False, "evidence_eligible": False}
  finally:
    wrapped.close()
  recovery = run_recovery(device="cpu", split="development", output=output / "diagnostic_recovery", scales=(4,), diagnostic=True)
  write_json(output / "diagnostic_c1.json", run_c1_cells("cpu", diagnostic=True))
  write_json(output / "diagnostic_retention.json", retention("cpu", diagnostic=True))
  result["real_collector_trials"] = len(recovery)
  return result


def execute(request):
  output = Path(request["output"])
  campaign = Path(request["campaign"])
  ident = identity(REPO, request["expected_sha"], local_check=request["local_check"])
  device = request["device"]
  if device == "cuda:0":
    lease = read_json(campaign / "gpu_budget.json").get("active")
    if not lease or lease.get("token") != request.get("lease_token"):
      raise ValueError("GPU worker has no matching supervisor budget lease")
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is unavailable; no CPU fallback for machine-room phases")
    torch.zeros(1, device=device)
  config = configuration()
  if config != read_json(REPO / "docs/experiments/artifacts/recovery_c1_pilot_v1/resolved_config.json"):
    raise ValueError("Resolved configuration drifted from preregistration")
  if PROTOCOL != read_json(REPO / "docs/experiments/artifacts/recovery_c1_pilot_v1/protocol.json"):
    raise ValueError("Protocol file differs from code contract")
  write_json(output / "resolved_config.json", config)
  versions = {n: importlib.metadata.version(n) for n in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")}
  locked = {x["name"]: x.get("version") for x in tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))["package"]}
  for name, version in versions.items():
    if locked.get(name) != version:
      raise ValueError(f"Installed {name}={version} differs from uv.lock={locked.get(name)}")
  if sys.version_info[:2] != (3, 11):
    raise ValueError("Frozen runtime requires Python 3.11")
  import mjlab
  if not Path(mjlab.__file__).resolve().is_relative_to((REPO.parent / "mjlab-main/src").resolve()):
    raise ValueError("Imported mjlab is not the verified sibling checkout")
  driver = None
  if device == "cuda:0":
    try:
      driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
      driver = "nvidia-smi unavailable; CUDA availability verified by allocation"
  runtime = {"python": sys.version, "nvidia_smi": driver, "device": device, "versions": versions,
             "gpu": torch.cuda.get_device_name(0) if device == "cuda:0" else None,
             "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device == "cuda:0" else None}
  write_json(output / "runtime.json", runtime)
  phase, seed = request["phase"], request["seed"]
  if request.get("viewer"):
    from mjlab.viewer import ViserPlayViewer
    checkpoint = campaign / f"train_seed{seed}/checkpoints/model_999.pt" if phase == "Evaluate" else None
    wrapped, policy, _ = open_env(device=device, seed=211, count=1, checkpoint=checkpoint)
    try:
      print("Viewer: open the printed local URL; enable command GUI before using sliders. Auto-stops at 3000 simulation steps.", flush=True)
      ViserPlayViewer(wrapped, policy).run(num_steps=3000)
    finally:
      wrapped.close()
    result = {"viewer_session_completed": True, "viewer_verdict": None, "user_feedback_required": True}
  elif phase == "Validate":
    smoke = cpu_smoke(output) if request.get("cpu_smoke") else None
    result = {"status": "PASS", "config_sha256": digest(config), "formal_training_started": False,
              "cpu_smoke": smoke, "baseline_eligible": False}
  elif phase == "Baseline":
    c1 = run_c1_cells(device)
    write_json(output / "c1.json", c1)
    # No recovery/training if the nominal classical system failed qualification.
    if not c1["flat_gate_passed"]:
      result = {"baseline_eligible": False, "failure": "C1_FLAT_QUALIFICATION_FAILED"}
    else:
      control = retention(device)
      write_json(output / "retention.json", control)
      rows = run_recovery(device=device, split="development", output=output / "recovery", scales=(4, 8))
      stats = summarize(rows)
      result = {"baseline_eligible": control["passed"] and all(x["successes"] == 128 and x["hard_failures"] == 0 for x in stats.values()),
                "recovery": stats, "retention_passed": control["passed"], "viewer_required": True}
  elif phase == "Train":
    require_approval(campaign, ident)
    wrapped, _, agent = open_env(device=device, seed=seed, count=256, training=True, campaign=campaign)
    try:
      runner = RecoveryPilotRunner(wrapped, dataclasses.asdict(agent), log_dir=str(output / "checkpoints"), device=device)
      runner.learn(1000)
      checkpoint = output / "checkpoints/model_999.pt"
      payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
      validate_checkpoint_record(payload["infos"][INFO_KEY], ident, seed, digest(config))
      result = {"completed_updates": 1000, "checkpoint": "checkpoints/model_999.pt",
                "checkpoint_sha256": file_sha(checkpoint), "training_seed": seed}
    finally:
      wrapped.close()
  elif phase == "Evaluate":
    require_approval(campaign, ident)
    source = campaign / f"train_seed{seed}"
    m = verify_sealed(source)
    from hoppertrex_mjlab.hybrid.recovery_pilot import require_identity
    require_identity(m, ident)
    checkpoint = source / "checkpoints/model_999.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_record(payload["infos"][INFO_KEY], ident, seed, digest(config))
    baseline = run_recovery(device=device, split="test", output=output / "baseline")
    candidate = run_recovery(device=device, split="test", output=output / "candidate", checkpoint=checkpoint)
    c1 = run_c1_cells(device, checkpoint)
    control = retention(device, checkpoint)
    write_json(output / "c1.json", c1)
    write_json(output / "retention.json", control)
    result = {"training_seed": seed, "comparison": compare(baseline, candidate),
              "retention_passed": control["passed"] and c1["flat_gate_passed"],
              "checkpoint_sha256": file_sha(checkpoint), "viewer_required": True}
  else:
    raise ValueError("Unsupported worker phase")
  write_json(output / "summary.json", result)
  return {**ident, "device": device, "phase": phase, "seed": seed,
          "config_sha256": digest(config), "scientific_pass": result.get("baseline_eligible")}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--request", type=Path, required=True)
  args = parser.parse_args()
  request = read_json(args.request)
  metadata = execute(request)
  write_json(Path(request["output"]) / "worker_result.json", metadata)


if __name__ == "__main__":
  main()
