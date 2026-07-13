"""Structured curriculum definitions shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


HYBRID_ACTION_NAMES = (
  "wheel_balance_residual",
  "wheel_yaw_residual",
  "left_thigh_residual",
  "right_thigh_residual",
  "left_knee_residual",
  "right_knee_residual",
)

# The balance policy is a residual around the qualified Stage0 controller.  A
# full-scale value must remain smaller than the nominal wheel targets so PPO
# cannot silently replace the controller during long training runs.
DEFAULT_ACTION_SCALES = (0.5, 1.0, 0.035, 0.035, 0.035, 0.035)
HYBRID_ACTION_STD = (0.15, 0.10, 0.05, 0.05, 0.05, 0.05)


@dataclass(frozen=True, kw_only=True)
class HybridStageCfg:
  """One Hybrid v2 capability stage with an invariant policy interface."""

  index: int
  capability: str
  action_mask: tuple[bool, ...]
  action_scales: tuple[float, ...]
  lin_vel_x_range: tuple[float, float]
  yaw_rate_range: tuple[float, float]
  posture_commands: bool
  randomization_level: int
  gate_suite: str
  push_interval_s: tuple[float, float] | None = None
  push_lin_vel_x: float = 0.0
  push_pitch_rate: float = 0.0

  def __post_init__(self) -> None:
    if len(self.action_mask) != 6 or len(self.action_scales) != 6:
      raise ValueError("Hybrid stages require exactly six action mask and scale values.")
    if any(scale < 0.0 for scale in self.action_scales):
      raise ValueError("Hybrid action scales must be non-negative.")
    if self.lin_vel_x_range[0] > self.lin_vel_x_range[1]:
      raise ValueError("lin_vel_x_range must be ordered.")
    if self.yaw_rate_range[0] > self.yaw_rate_range[1]:
      raise ValueError("yaw_rate_range must be ordered.")
    if self.randomization_level not in (0, 1, 2):
      raise ValueError("randomization_level must be 0, 1, or 2.")
    if self.push_interval_s is not None:
      if self.push_interval_s[0] <= 0.0 or self.push_interval_s[0] > self.push_interval_s[1]:
        raise ValueError("push_interval_s must be positive and ordered.")

  @property
  def action_dim(self) -> int:
    return len(HYBRID_ACTION_NAMES)


def _stage(
  index: int,
  capability: str,
  mask: tuple[bool, bool, bool, bool, bool, bool],
  lin_range: tuple[float, float],
  yaw_range: tuple[float, float],
  *,
  posture: bool,
  randomization_level: int = 0,
  gate_suite: str,
  push_interval_s: tuple[float, float] | None = None,
  push_lin_vel_x: float = 0.0,
  push_pitch_rate: float = 0.0,
) -> HybridStageCfg:
  return HybridStageCfg(
    index=index,
    capability=capability,
    action_mask=mask,
    action_scales=DEFAULT_ACTION_SCALES,
    lin_vel_x_range=lin_range,
    yaw_rate_range=yaw_range,
    posture_commands=posture,
    randomization_level=randomization_level,
    gate_suite=gate_suite,
    push_interval_s=push_interval_s,
    push_lin_vel_x=push_lin_vel_x,
    push_pitch_rate=push_pitch_rate,
  )


HYBRID_STAGES = {
  0: _stage(
    0,
    "controller qualification",
    (False, False, False, False, False, False),
    (-0.07, 0.07),
    (0.0, 0.0),
    posture=False,
    gate_suite="controller",
  ),
  1: _stage(
    1,
    "fixed-leg residual robustness and transient compensation",
    (True, False, False, False, False, False),
    (-0.07, 0.07),
    (0.0, 0.0),
    posture=False,
    randomization_level=1,
    gate_suite="residual",
    push_interval_s=(5.0, 8.0),
    push_lin_vel_x=0.04,
    push_pitch_rate=0.06,
  ),
  2: _stage(
    2,
    "fixed-leg planar velocity",
    (True, True, False, False, False, False),
    (-0.07, 0.07),
    (-0.10, 0.10),
    posture=False,
    gate_suite="planar",
  ),
  3: _stage(
    3,
    "leg posture control",
    (True, True, True, True, True, True),
    (0.0, 0.0),
    (0.0, 0.0),
    posture=True,
    gate_suite="posture",
  ),
  4: _stage(
    4,
    "integrated wheel-leg control",
    (True, True, True, True, True, True),
    (-0.07, 0.07),
    (-0.10, 0.10),
    posture=True,
    gate_suite="integrated",
  ),
  5: _stage(
    5,
    "robust wheel-leg control",
    (True, True, True, True, True, True),
    (-0.07, 0.07),
    (-0.10, 0.10),
    posture=True,
    randomization_level=2,
    gate_suite="robust",
    push_interval_s=(3.0, 5.0),
    push_lin_vel_x=0.08,
    push_pitch_rate=0.12,
  ),
}
