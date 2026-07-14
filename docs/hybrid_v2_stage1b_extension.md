# Hybrid v2 Stage1-B extension protocol

Stage1-B is a **same-stage curriculum continuation** from the screened
Stage1-A residual checkpoint. It does not rerun Stage0/LQR, does not create
Stage2, and does not ask PPO to relearn low-speed standing, forward, or reverse
motion. Those are the controller baseline's responsibility.

The Stage1-B residual is evaluated only as an ablation against the same
zero-residual LQR. Its purpose is to improve the controller's margin under
speed extension, mild model mismatch, kicks, and command transitions while
preserving nominal behavior.

## Training profile

- commands: 20% standing, 60% balanced-sign nominal `|v_x| in [0.03, 0.07]`,
  20% balanced-sign extension `|v_x| in [0.07, 0.10]` m/s;
- 50% of training worlds receive one fixed startup mismatch, shared by both
  wheels in a world: chassis mass/inertia `+/-5%`, chassis COM x/z `+/-5 mm`,
  wheel friction `+/-10%`, wheel radius `+/-2%`, and wheel velocity-actuator
  gain `+/-5%`;
- the remaining 50% are nominal; mismatch is neither reset each episode nor
  independently sampled left/right;
- formal gate: `v_x = -0.10, -0.07, 0, +0.07, +0.10`, kick recovery,
  command transitions, and a deterministic mismatch group. It requires at
  least 32 environments and 3000 steps.

The 100-iteration probe and its screen are engineering checks only. A passing
single seed is not a three-seed research result and does not authorize Stage2
or a longer run by itself.

## Required handoff

Use the exact Stage1-A checkpoint that produced the passing residual screen.
The handoff utility verifies its bootstrap/training provenance and verifies
that the screen JSON's checkpoint SHA256 is the same file. It retains the
policy weights, clears optimizer moments, resets checkpoint iteration to zero,
and records a `hybrid_stage1_extension` provenance record.

```powershell
$python = ".\.venv\Scripts\python.exe"
$source = "<absolute path to the passing Stage1-A model.pt>"
$screen = "<absolute path to its passing seed1 screen JSON>"
$handoffRun = "hybrid_v2_stage1b_handoff_seed1"
$handoffRoot = "src\hoppertrex_mjlab\logs\rsl_rl\hoppertrex_balance\$handoffRun"
$handoff = "$handoffRoot\model_0.pt"

& $python -m hoppertrex_mjlab.scripts.rsl_rl.prepare_hybrid_stage1_extension `
  --source-checkpoint $source `
  --source-gate-json $screen `
  --output-checkpoint $handoff
```

Do not use `--reset-collapsed-active-std` unless the utility explicitly reports
that the balance-residual standard deviation has collapsed and restoring it is
an intentional, recorded decision.

## One bounded probe

```powershell
& $python -m hoppertrex_mjlab.scripts.rsl_rl.train `
  HopperTrex-Hybrid-v2-Stage1 `
  --env.scene.num-envs 256 `
  --agent.max-iterations 100 `
  --agent.save-interval 25 `
  --agent.seed 1 `
  --agent.resume True `
  --agent.load-run ".*hybrid_v2_stage1b_handoff_seed1.*" `
  --agent.load-checkpoint "model_0.pt" `
  --agent.run-name "hybrid_v2_stage1b_probe_seed1"
```

Then locate the newest `model_*.pt` in the probe run and screen it at 16
environments/1000 steps. Only if the screen passes should it be reviewed in
Viser and then evaluated by the single-seed formal gate at 32 environments and
3000 steps. Do not commit or push from a machine-room checkout.

## Retention-protected Stage1-B to Stage2 handoff

A Stage1-B checkpoint may enter Stage2 only when its exact SHA256 appears in a
passing live formal Stage1 residual envelope. `migrate_hybrid_stage.py` requires
that envelope through `--source-gate-json`; it rejects screen results, offline
scenario files, stale mismatch profiles, mismatched training revisions, and a
gate produced by another checkpoint.

Stage2 activates only the yaw residual head. The migrated checkpoint preserves
the Stage1 balance head, zeros the new yaw output row, resets the new yaw action
standard deviation to `0.10`, clears optimizer moments, and records the source
checkpoint and gate hashes. If the Stage1 balance-head std is collapsed, the
migration stops unless the reset is explicitly authorized.

Stage2 training rehearses Stage1 rather than relying only on a post-training
test:

- 10% standing, 25% linear-only, 25% yaw-only, 30% nominal combined commands;
- 10% linear-only Stage1-B retention commands with
  `|v_x| in [0.07, 0.10] m/s` and zero yaw;
- the Stage1 level-1 reset disturbance, mild fixed startup mismatch, and
  `5-8 s` kicks with `x +/-0.04 m/s` and pitch rate `+/-0.06 rad/s`;
- a healthy-state penalty only on balance residual index 0, leaving the new yaw
  head free to learn;
- no legacy pure-PPO linear sign reward.

After a bounded 100-iteration Stage2 seed-1 probe, use the same Stage2
checkpoint twice. First run the Stage1 residual suite to produce a retention
envelope. Then run the Stage2 planar suite with that envelope supplied through
`--stage1-retention-file`. The retention file must match profile, seed, git
revision, checkpoint SHA256, and Stage1-B mismatch profile. Screen is an
engineering rejection check only; formal Stage2 requires a matching formal
retention result. A Stage2 failure does not revoke the already frozen Stage1-B
result and does not authorize more Stage1 training.
