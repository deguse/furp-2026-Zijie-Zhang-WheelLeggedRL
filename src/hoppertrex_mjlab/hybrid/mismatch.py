"""Mild, sagittally symmetric Stage1 model mismatch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


STAGE1_MISMATCH_PROFILE_VERSION = "stage1b_speed010_mild_v1"
STAGE1_MISMATCH_FRACTION = 0.50
STAGE1_MASS_INERTIA_SCALE_RANGE = (0.95, 1.05)
STAGE1_COM_X_RANGE_M = (-0.005, 0.005)
STAGE1_COM_Z_RANGE_M = (-0.005, 0.005)
STAGE1_WHEEL_FRICTION_SCALE_RANGE = (0.90, 1.10)
STAGE1_WHEEL_RADIUS_SCALE_RANGE = (0.98, 1.02)
STAGE1_WHEEL_ACTUATOR_GAIN_SCALE_RANGE = (0.95, 1.05)


@dataclass(frozen=True)
class Stage1MismatchAudit:
  """Actual per-world mismatch values used by training or evaluation."""

  is_mismatch: torch.Tensor
  mass_inertia_scale: torch.Tensor
  com_x_offset_m: torch.Tensor
  com_z_offset_m: torch.Tensor
  wheel_friction_scale: torch.Tensor
  wheel_radius_scale: torch.Tensor
  wheel_actuator_gain_scale: torch.Tensor

  def select(self, env_ids: torch.Tensor) -> dict[str, list[float] | list[bool]]:
    ids = env_ids.to(device=self.is_mismatch.device, dtype=torch.long)
    return {
      "is_mismatch": self.is_mismatch[ids].detach().cpu().tolist(),
      "mass_inertia_scale": (
        self.mass_inertia_scale[ids].detach().cpu().tolist()
      ),
      "com_x_offset_m": self.com_x_offset_m[ids].detach().cpu().tolist(),
      "com_z_offset_m": self.com_z_offset_m[ids].detach().cpu().tolist(),
      "wheel_friction_scale": (
        self.wheel_friction_scale[ids].detach().cpu().tolist()
      ),
      "wheel_radius_scale": (
        self.wheel_radius_scale[ids].detach().cpu().tolist()
      ),
      "wheel_actuator_gain_scale": (
        self.wheel_actuator_gain_scale[ids].detach().cpu().tolist()
      ),
    }


def stage1_mismatch_spec() -> dict[str, object]:
  return {
    "profile_version": STAGE1_MISMATCH_PROFILE_VERSION,
    "mismatch_fraction": STAGE1_MISMATCH_FRACTION,
    "mass_inertia_scale_range": list(STAGE1_MASS_INERTIA_SCALE_RANGE),
    "com_x_offset_m_range": list(STAGE1_COM_X_RANGE_M),
    "com_z_offset_m_range": list(STAGE1_COM_Z_RANGE_M),
    "wheel_friction_scale_range": list(STAGE1_WHEEL_FRICTION_SCALE_RANGE),
    "wheel_radius_scale_range": list(STAGE1_WHEEL_RADIUS_SCALE_RANGE),
    "wheel_actuator_gain_scale_range": list(
      STAGE1_WHEEL_ACTUATOR_GAIN_SCALE_RANGE
    ),
    "left_right_shared": True,
  }


def _ids_tensor(ids: Sequence[int] | slice, count: int, device: str) -> torch.Tensor:
  if isinstance(ids, slice):
    return torch.arange(count, device=device, dtype=torch.long)[ids]
  return torch.tensor(ids, device=device, dtype=torch.long)


def _sample(
  value_range: tuple[float, float],
  count: int,
  device: str,
) -> torch.Tensor:
  low, high = value_range
  return torch.empty(count, device=device).uniform_(low, high)


def _selected_mismatch_ids(
  env_ids: torch.Tensor,
  *,
  mismatch_fraction: float,
  group_count: int | None,
  mismatch_group_indices: tuple[int, ...],
) -> torch.Tensor:
  if group_count is not None:
    if group_count <= 0:
      raise ValueError("group_count must be positive.")
    if not mismatch_group_indices:
      raise ValueError("mismatch_group_indices cannot be empty with group_count.")
    selected = torch.zeros_like(env_ids, dtype=torch.bool)
    remainders = env_ids.remainder(group_count)
    for group_index in mismatch_group_indices:
      if group_index < 0 or group_index >= group_count:
        raise ValueError("Mismatch group index is outside group_count.")
      selected |= remainders == group_index
    return env_ids[selected]
  if not 0.0 <= mismatch_fraction <= 1.0:
    raise ValueError("mismatch_fraction must be in [0, 1].")
  return env_ids[torch.rand(len(env_ids), device=env_ids.device) < mismatch_fraction]


@requires_model_fields(
  "body_mass",
  "body_inertia",
  "body_ipos",
  "geom_friction",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "actuator_gainprm",
  "actuator_biasprm",
  recompute=RecomputeLevel.set_const,
)
def apply_stage1_mild_mismatch(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  chassis_cfg: SceneEntityCfg,
  wheel_geom_cfg: SceneEntityCfg,
  wheel_actuator_cfg: SceneEntityCfg,
  mismatch_fraction: float = STAGE1_MISMATCH_FRACTION,
  group_count: int | None = None,
  mismatch_group_indices: tuple[int, ...] = (),
) -> None:
  """Apply fixed per-world mismatch while preserving left/right symmetry."""

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  mismatch_ids = _selected_mismatch_ids(
    env_ids,
    mismatch_fraction=mismatch_fraction,
    group_count=group_count,
    mismatch_group_indices=mismatch_group_indices,
  )

  audit_values = {
    "is_mismatch": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    "mass_inertia_scale": torch.ones(env.num_envs, device=env.device),
    "com_x_offset_m": torch.zeros(env.num_envs, device=env.device),
    "com_z_offset_m": torch.zeros(env.num_envs, device=env.device),
    "wheel_friction_scale": torch.ones(env.num_envs, device=env.device),
    "wheel_radius_scale": torch.ones(env.num_envs, device=env.device),
    "wheel_actuator_gain_scale": torch.ones(env.num_envs, device=env.device),
  }
  if len(mismatch_ids) == 0:
    env._hoppertrex_stage1_mismatch = Stage1MismatchAudit(**audit_values)
    return

  robot = env.scene[chassis_cfg.name]
  chassis_ids = _ids_tensor(
    chassis_cfg.body_ids,
    robot.num_bodies,
    env.device,
  )
  wheel_geom_ids = _ids_tensor(
    wheel_geom_cfg.geom_ids,
    robot.num_geoms,
    env.device,
  )
  if len(chassis_ids) != 1:
    raise ValueError("Stage1 mismatch requires exactly one chassis body.")
  if len(wheel_geom_ids) != 2:
    raise ValueError("Stage1 mismatch requires exactly two wheel geoms.")

  count = len(mismatch_ids)
  mass_scale = _sample(STAGE1_MASS_INERTIA_SCALE_RANGE, count, env.device)
  com_x = _sample(STAGE1_COM_X_RANGE_M, count, env.device)
  com_z = _sample(STAGE1_COM_Z_RANGE_M, count, env.device)
  friction_scale = _sample(
    STAGE1_WHEEL_FRICTION_SCALE_RANGE,
    count,
    env.device,
  )
  radius_scale = _sample(STAGE1_WHEEL_RADIUS_SCALE_RANGE, count, env.device)
  actuator_scale = _sample(
    STAGE1_WHEEL_ACTUATOR_GAIN_SCALE_RANGE,
    count,
    env.device,
  )

  model = env.sim.model
  default_mass = env.sim.get_default_field("body_mass")
  default_inertia = env.sim.get_default_field("body_inertia")
  default_ipos = env.sim.get_default_field("body_ipos")
  body_grid = mismatch_ids[:, None], chassis_ids[None, :]
  model.body_mass[body_grid] = default_mass[chassis_ids][None, :] * mass_scale[:, None]
  model.body_inertia[body_grid] = (
    default_inertia[chassis_ids][None, :, :] * mass_scale[:, None, None]
  )
  model.body_ipos[body_grid] = default_ipos[chassis_ids][None, :, :]
  model.body_ipos[mismatch_ids, chassis_ids[0], 0] += com_x
  model.body_ipos[mismatch_ids, chassis_ids[0], 2] += com_z

  default_friction = env.sim.get_default_field("geom_friction")
  default_size = env.sim.get_default_field("geom_size")
  geom_grid = mismatch_ids[:, None], wheel_geom_ids[None, :]
  model.geom_friction[geom_grid] = default_friction[wheel_geom_ids][None, :, :]
  model.geom_friction[mismatch_ids[:, None], wheel_geom_ids[None, :], 0] *= (
    friction_scale[:, None]
  )
  model.geom_size[geom_grid] = default_size[wheel_geom_ids][None, :, :]
  model.geom_size[mismatch_ids[:, None], wheel_geom_ids[None, :], 0] *= (
    radius_scale[:, None]
  )
  _recompute_geom_bounds(
    env,
    mismatch_ids.to(dtype=torch.int),
    wheel_geom_cfg,
  )

  # SceneEntityCfg actuator ids index individual MuJoCo control rows, whereas
  # robot.actuators groups implementation objects (leg PD and wheel velocity).
  # Do not use the former as indices into the latter: wheel ids are [2, 5],
  # while the implementation list has length two.
  ctrl_ids = _ids_tensor(
    wheel_actuator_cfg.actuator_ids,
    robot.num_actuators,
    env.device,
  )
  if len(ctrl_ids) != 2:
    raise ValueError("Stage1 mismatch requires exactly two wheel actuator controls.")
  default_gain = env.sim.get_default_field("actuator_gainprm")
  default_bias = env.sim.get_default_field("actuator_biasprm")
  actuator_grid = mismatch_ids[:, None], ctrl_ids[None, :]
  model.actuator_gainprm[actuator_grid] = default_gain[ctrl_ids][None, :, :]
  model.actuator_biasprm[actuator_grid] = default_bias[ctrl_ids][None, :, :]
  model.actuator_gainprm[
    mismatch_ids[:, None], ctrl_ids[None, :], 0
  ] *= actuator_scale[:, None]
  model.actuator_biasprm[
    mismatch_ids[:, None], ctrl_ids[None, :], 2
  ] *= actuator_scale[:, None]

  audit_values["is_mismatch"][mismatch_ids] = True
  for name, values in (
    ("mass_inertia_scale", mass_scale),
    ("com_x_offset_m", com_x),
    ("com_z_offset_m", com_z),
    ("wheel_friction_scale", friction_scale),
    ("wheel_radius_scale", radius_scale),
    ("wheel_actuator_gain_scale", actuator_scale),
  ):
    audit_values[name][mismatch_ids] = values
  env._hoppertrex_stage1_mismatch = Stage1MismatchAudit(**audit_values)


def stage1_mismatch_audit(env: ManagerBasedRlEnv) -> Stage1MismatchAudit:
  audit = getattr(env, "_hoppertrex_stage1_mismatch", None)
  if not isinstance(audit, Stage1MismatchAudit):
    raise RuntimeError("Stage1 mismatch audit is unavailable on this environment.")
  return audit


__all__ = [
  "STAGE1_MISMATCH_FRACTION",
  "STAGE1_MISMATCH_PROFILE_VERSION",
  "Stage1MismatchAudit",
  "apply_stage1_mild_mismatch",
  "stage1_mismatch_audit",
  "stage1_mismatch_spec",
]
