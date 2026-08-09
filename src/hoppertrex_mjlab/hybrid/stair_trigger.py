"""CTBC-style contact trigger for the residual stair camp (mainline doc S5B).

The camp drives its `stair_mode` flag from a fixed classical threshold on the
contact force. It deliberately does NOT use the repo's classical stair FSM
(`stair_classical.stair_controller_step`), whose CLIMB transition is gated by
the `ContactDetectorCfg` three-channel threshold family that C2-j3 falsified;
S5B Protocol 1 pins `stair_maneuver = None` for exactly that reason.

Metric (frozen, S5B Protocol 2): ``|F0 * nx|`` where ``F0`` is the
contact-frame force component 0 (the normal force, per mjlab's contact sensor
field layout) and ``nx`` is the global-x component of the contact normal,
maximised over the sensor's contact slots and zeroed on slots reporting no
contact. This is the same metric that was replayed over the frozen C2-j3
formal NPZs, where the 18 N x 3-frame rule detected 288/288 impact pairs
(100%) with zero pre-impact fires and a weakest-pair run of 4 ticks against
the 3-tick window (experiment log S3.57).

Threshold and window are preregistered decisions, [U]-confirmed 2026-08-04.
18 N is the only cross-18-cell zero-false-positive threshold measured under
C2 conditions (flat max 17.63 N, so the FP margin is only 1.02x); the
envelope-transfer validity gate in S5B Protocol 2 requires a fresh zero-FP
check under camp randomization before the first reported training run.
"""

from __future__ import annotations

import torch

# Preregistered trigger constants (S5B decisions table, [U] 2026-08-04).
STAIR_TRIGGER_FORCE_N = 18.0
STAIR_TRIGGER_WINDOW = 3

# The camp adds its own contact sensor rather than reusing the termination
# sensor: the trigger needs the per-slot contact frame force and the global
# normal, which the non-wheel termination sensor does not carry.
STAIR_TRIGGER_SENSOR_NAME = "stair_camp_wheel_contact_trigger"
STAIR_TRIGGER_SENSOR_FIELDS = ("found", "force", "normal")
# Matches the C2-j3 capture that produced the frozen evidence: 8 slots per
# wheel primary, so a wheel resting across a riser edge cannot silently drop
# the riser contact in favour of the ground contact.
STAIR_TRIGGER_SLOTS_PER_WHEEL = 8


def stair_trigger_metric(
  *,
  found: torch.Tensor,
  force_contact_frame: torch.Tensor,
  normal_global: torch.Tensor,
) -> torch.Tensor:
  """Reduce one contact-sensor sample to the frozen per-env trigger metric.

  Args:
    found: ``[B, N]`` contact counts; a slot counts only when it is > 0.
    force_contact_frame: ``[B, N, 3]`` contact-frame force.
    normal_global: ``[B, N, 3]`` global contact normal.

  Returns:
    ``[B]`` maximum of ``|F0 * nx|`` over the slots that report contact.
  """

  expected = (*found.shape, 3)
  for name, value in (
    ("force_contact_frame", force_contact_frame),
    ("normal_global", normal_global),
  ):
    if tuple(value.shape) != expected:
      raise ValueError(f"{name} must have shape {expected}.")
  magnitude = (force_contact_frame[..., 0] * normal_global[..., 0]).abs()
  return torch.where(
    found > 0, magnitude, torch.zeros_like(magnitude)
  ).amax(dim=-1)


def update_stair_trigger(
  *,
  latched: torch.Tensor,
  streak: torch.Tensor,
  metric: torch.Tensor,
  threshold: float = STAIR_TRIGGER_FORCE_N,
  window: int = STAIR_TRIGGER_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Advance the sliding-window trigger by one control step.

  The counter resets on any sample below threshold, so the window is
  strictly consecutive. The latch is monotone within an episode: S5B
  Protocol 2 registers `stair_mode` as latching ON once triggered and
  resetting only at episode reset, so there is no mid-episode exit path.

  Args:
    latched: ``[B]`` bool `stair_mode` state carried from the previous step.
    streak: ``[B]`` integer count of consecutive above-threshold samples.
    metric: ``[B]`` current trigger metric.
    threshold: force threshold in newtons.
    window: number of consecutive samples required to fire.

  Returns:
    ``(latched, streak)`` for the current step. The returned `latched`
    already includes a fire on this step, so a rising edge is visible to
    every consumer within the same control step.
  """

  if window < 1:
    raise ValueError("Stair trigger window must be at least one sample.")
  hit = metric >= threshold
  # Saturating counter: it only has to distinguish "< window" from ">= window",
  # and clamping keeps it bounded over long episodes.
  advanced = (streak + 1) * hit
  clamped = torch.clamp(advanced, max=window)
  return latched | (clamped >= window), clamped
