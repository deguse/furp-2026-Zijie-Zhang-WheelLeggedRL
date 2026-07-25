"""Discrete wheel-balance identification and honest LQR/PD qualification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_are


CONTROLLER_STATE_NAMES = (
  "pitch",
  "pitch_rate",
  "vx_error",
  "signed_wheel_speed_error",
)

# The controller state is built from runtime conventions that the gain hash
# does not cover: the wheel radius used for the desired wheel speed, the
# pitch convention atan2(gravity_x, -gravity_z), and the signed wheel speed
# 0.5 * (right - left). Artifacts record this block so the task loader can
# reject a gain identified against a different state construction.
STATE_DEFINITION_VERSION = "hybrid_v2_state_v1"
NOMINAL_WHEEL_RADIUS_M = 0.100


def state_construction_spec(wheel_radius: float) -> dict[str, object]:
  """State-provenance block recorded in artifacts and checked at load."""

  radius = float(wheel_radius)
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("wheel_radius must be positive and finite.")
  return {
    "state_definition_version": STATE_DEFINITION_VERSION,
    "wheel_radius": radius,
  }


@dataclass(frozen=True)
class DiscreteModel:
  a: NDArray[np.float64]
  b: NDArray[np.float64]


@dataclass(frozen=True)
class NrmseResult:
  by_state: NDArray[np.float64]
  maximum: float


@dataclass(frozen=True)
class ControllerDesign:
  controller_type: str
  gain: NDArray[np.float64]
  model: DiscreteModel
  controllability_rank: int
  heldout_nrmse: NrmseResult
  fallback_reasons: tuple[str, ...]

  @property
  def gain_hash(self) -> str:
    payload = {
      "controller_type": self.controller_type,
      "state_names": CONTROLLER_STATE_NAMES,
      "gain": self.gain.tolist(),
    }
    encoded = json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AffineEquilibrium:
  """Steady state/input pair used to identify local delta dynamics."""

  state: NDArray[np.float64]
  input: NDArray[np.float64]


@dataclass(frozen=True)
class IncumbentAnchoredGains:
  """One globally blended gain grid with auditable damping margins."""

  alpha: float
  gains: NDArray[np.float64]
  raw_spectral_radius: tuple[float, ...]
  incumbent_spectral_radius: tuple[float, ...]
  deployed_spectral_radius: tuple[float, ...]


def controller_design_to_dict(design: ControllerDesign) -> dict[str, object]:
  """Convert a qualified controller design to auditable JSON data."""

  nrmse_by_state = {
    name: float(value)
    for name, value in zip(
      CONTROLLER_STATE_NAMES,
      design.heldout_nrmse.by_state,
      strict=True,
    )
  }
  return {
    "schema_version": 1,
    "controller_type": design.controller_type,
    "state_names": list(CONTROLLER_STATE_NAMES),
    "model": {
      "a": design.model.a.tolist(),
      "b": design.model.b.tolist(),
    },
    "gain": design.gain.tolist(),
    "controllability_rank": design.controllability_rank,
    "heldout_one_step_nrmse": {
      "by_state": nrmse_by_state,
      "maximum": design.heldout_nrmse.maximum,
    },
    "fallback_reasons": list(design.fallback_reasons),
    "gain_hash": design.gain_hash,
  }


def _samples(name: str, value: ArrayLike) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 2 or array.shape[0] < 1:
    raise ValueError(f"{name} must be a non-empty two-dimensional array.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values.")
  return array


def estimate_affine_equilibrium(
  states: ArrayLike,
  inputs: ArrayLike,
) -> AffineEquilibrium:
  """Estimate the steady affine point from a zero-residual warmup window."""

  x = _samples("equilibrium states", states)
  u = _samples("equilibrium inputs", inputs)
  if x.shape[0] != u.shape[0]:
    raise ValueError("Equilibrium states and inputs must have equal row counts.")
  if x.shape[1] != len(CONTROLLER_STATE_NAMES) or u.shape[1] != 1:
    raise ValueError("Hybrid affine equilibrium requires four states and one input.")
  return AffineEquilibrium(
    state=np.mean(x, axis=0),
    input=np.mean(u, axis=0),
  )


def center_affine_transitions(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
  equilibrium: AffineEquilibrium,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Return delta transitions around one frozen state/input equilibrium."""

  x = _samples("states", states)
  u = _samples("inputs", inputs)
  x_next = _samples("next_states", next_states)
  if x.shape != x_next.shape or x.shape[0] != u.shape[0]:
    raise ValueError("Affine transition arrays must have compatible shapes.")
  state = np.asarray(equilibrium.state, dtype=np.float64)
  control = np.asarray(equilibrium.input, dtype=np.float64)
  if state.shape != (x.shape[1],) or control.shape != (u.shape[1],):
    raise ValueError("Affine equilibrium dimensions do not match transitions.")
  if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
    raise ValueError("Affine equilibrium must contain only finite values.")
  return x - state, u - control, x_next - state


def closed_loop_spectral_radius(
  a: ArrayLike,
  b: ArrayLike,
  gain: ArrayLike,
) -> float:
  """Return rho(A-BK) for the runtime convention u=-Kx."""

  a_array = np.asarray(a, dtype=np.float64)
  b_array = np.asarray(b, dtype=np.float64)
  gain_array = np.asarray(gain, dtype=np.float64)
  if a_array.ndim != 2 or a_array.shape[0] != a_array.shape[1]:
    raise ValueError("A must be square.")
  if b_array.shape[0] != a_array.shape[0]:
    raise ValueError("B row count must match A.")
  if gain_array.shape != (b_array.shape[1], a_array.shape[0]):
    raise ValueError("Gain shape must be (input_dim, state_dim).")
  radius = float(np.max(np.abs(np.linalg.eigvals(a_array - b_array @ gain_array))))
  if not math.isfinite(radius):
    raise ValueError("Closed-loop spectral radius must be finite.")
  return radius


def anchor_gains_to_incumbent(
  models: Sequence[DiscreteModel],
  proposed_gains: ArrayLike,
  incumbent_gain: ArrayLike,
  *,
  spectral_margin: float = 0.01,
  alpha_grid: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0),
) -> IncumbentAnchoredGains:
  """Choose the largest global proposal blend with no damping regression."""

  if not models:
    raise ValueError("At least one identified model is required.")
  if not math.isfinite(spectral_margin) or spectral_margin < 0.0:
    raise ValueError("spectral_margin must be finite and non-negative.")
  proposals = np.asarray(proposed_gains, dtype=np.float64)
  incumbent = np.asarray(incumbent_gain, dtype=np.float64)
  expected = (len(models), 1, models[0].a.shape[0])
  if proposals.shape != expected:
    raise ValueError(f"proposed_gains must have shape {expected}.")
  if incumbent.shape == (expected[2],):
    incumbent = incumbent.reshape(1, expected[2])
  if incumbent.shape != (1, expected[2]):
    raise ValueError("incumbent_gain must have shape (1, state_dim).")
  alphas = tuple(float(value) for value in alpha_grid)
  if (
    not alphas
    or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in alphas)
    or tuple(sorted(alphas, reverse=True)) != alphas
    or alphas[-1] != 0.0
  ):
    raise ValueError("alpha_grid must descend from [0, 1] and end at zero.")
  incumbent_radii = tuple(
    closed_loop_spectral_radius(model.a, model.b, incumbent) for model in models
  )
  raw_radii = tuple(
    closed_loop_spectral_radius(model.a, model.b, proposals[index])
    for index, model in enumerate(models)
  )
  for alpha in alphas:
    deployed = (1.0 - alpha) * incumbent[None, :, :] + alpha * proposals
    deployed_radii = tuple(
      closed_loop_spectral_radius(model.a, model.b, deployed[index])
      for index, model in enumerate(models)
    )
    if all(
      deployed_radius <= incumbent_radius + spectral_margin
      for deployed_radius, incumbent_radius in zip(
        deployed_radii, incumbent_radii, strict=True
      )
    ):
      return IncumbentAnchoredGains(
        alpha=alpha,
        gains=deployed,
        raw_spectral_radius=raw_radii,
        incumbent_spectral_radius=incumbent_radii,
        deployed_spectral_radius=deployed_radii,
      )
  raise RuntimeError("Zero incumbent blend unexpectedly failed its own damping gate.")


def fit_discrete_model(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
) -> DiscreteModel:
  """Fit ``x[k+1] = A x[k] + B u[k]`` by least squares."""

  x = _samples("states", states)
  u = _samples("inputs", inputs)
  x_next = _samples("next_states", next_states)
  if x.shape != x_next.shape:
    raise ValueError("states and next_states must have identical shapes.")
  if x.shape[0] != u.shape[0]:
    raise ValueError("states and inputs must contain the same number of samples.")
  if x.shape[1] != len(CONTROLLER_STATE_NAMES) or u.shape[1] != 1:
    raise ValueError("Hybrid wheel identification requires four states and one input.")

  coefficients, _, _, _ = np.linalg.lstsq(
    np.concatenate((x, u), axis=1),
    x_next,
    rcond=None,
  )
  state_dim = x.shape[1]
  return DiscreteModel(
    a=coefficients[:state_dim, :].T,
    b=coefficients[state_dim:, :].T,
  )


def controllability_rank(
  a: ArrayLike,
  b: ArrayLike,
  *,
  absolute_tolerance: float = 1.0e-10,
) -> int:
  a_array = np.asarray(a, dtype=np.float64)
  b_array = np.asarray(b, dtype=np.float64)
  if a_array.ndim != 2 or a_array.shape[0] != a_array.shape[1]:
    raise ValueError("A must be square.")
  if b_array.ndim != 2 or b_array.shape[0] != a_array.shape[0]:
    raise ValueError("B row count must match A.")
  blocks = [b_array]
  for _ in range(1, a_array.shape[0]):
    blocks.append(a_array @ blocks[-1])
  singular_values = np.linalg.svd(np.concatenate(blocks, axis=1), compute_uv=False)
  relative_tolerance = (
    max(a_array.shape)
    * np.finfo(np.float64).eps
    * max(float(singular_values[0]), 1.0)
  )
  tolerance = max(absolute_tolerance, relative_tolerance)
  return int(np.count_nonzero(singular_values > tolerance))


def one_step_nrmse(actual: ArrayLike, predicted: ArrayLike) -> NrmseResult:
  """Return range-normalized RMSE for each state and the worst state."""

  actual_array = _samples("actual", actual)
  predicted_array = _samples("predicted", predicted)
  if actual_array.shape != predicted_array.shape:
    raise ValueError("actual and predicted arrays must have identical shapes.")
  rmse = np.sqrt(np.mean(np.square(predicted_array - actual_array), axis=0))
  dynamic_range = np.ptp(actual_array, axis=0)
  by_state = np.empty_like(rmse)
  constant = dynamic_range <= np.finfo(np.float64).eps
  by_state[~constant] = rmse[~constant] / dynamic_range[~constant]
  by_state[constant] = np.where(
    rmse[constant] <= np.finfo(np.float64).eps,
    0.0,
    np.inf,
  )
  return NrmseResult(by_state=by_state, maximum=float(np.max(by_state)))


def solve_lqr_gain(
  a: ArrayLike,
  b: ArrayLike,
  q: ArrayLike,
  r: ArrayLike,
) -> NDArray[np.float64]:
  """Solve the discrete algebraic Riccati equation for ``u = -Kx``."""

  a_array = np.asarray(a, dtype=np.float64)
  b_array = np.asarray(b, dtype=np.float64)
  q_array = np.asarray(q, dtype=np.float64)
  r_array = np.asarray(r, dtype=np.float64)
  p = solve_discrete_are(a_array, b_array, q_array, r_array)
  return np.linalg.solve(
    b_array.T @ p @ b_array + r_array,
    b_array.T @ p @ a_array,
  )


def identify_controller(
  states: ArrayLike,
  inputs: ArrayLike,
  next_states: ArrayLike,
  *,
  heldout_states: ArrayLike,
  heldout_inputs: ArrayLike,
  heldout_next_states: ArrayLike,
  q_diag: Sequence[float],
  r_diag: Sequence[float],
  pd_gain: Sequence[float],
  nrmse_limit: float = 0.15,
) -> ControllerDesign:
  """Fit and qualify an LQR, or return an explicitly labelled PD fallback."""

  if nrmse_limit <= 0.0:
    raise ValueError("nrmse_limit must be positive.")
  model = fit_discrete_model(states, inputs, next_states)
  heldout_x = _samples("heldout_states", heldout_states)
  heldout_u = _samples("heldout_inputs", heldout_inputs)
  heldout_next = _samples("heldout_next_states", heldout_next_states)
  if heldout_x.shape != heldout_next.shape or heldout_x.shape[0] != heldout_u.shape[0]:
    raise ValueError("Held-out arrays must have compatible sample shapes.")
  prediction = heldout_x @ model.a.T + heldout_u @ model.b.T
  nrmse = one_step_nrmse(heldout_next, prediction)
  rank = controllability_rank(model.a, model.b)
  state_dim = model.a.shape[0]

  reasons: list[str] = []
  if rank != state_dim:
    reasons.append(f"controllability rank {rank} is below state dimension {state_dim}")
  if nrmse.maximum > nrmse_limit:
    reasons.append(
      f"held-out one-step NRMSE {nrmse.maximum:.6f} exceeds {nrmse_limit:.6f}"
    )

  gain: NDArray[np.float64]
  if not reasons:
    q_values = np.asarray(q_diag, dtype=np.float64)
    r_values = np.asarray(r_diag, dtype=np.float64)
    if q_values.shape != (state_dim,) or r_values.shape != (model.b.shape[1],):
      raise ValueError("Q and R diagonals must match model state and input dimensions.")
    try:
      gain = solve_lqr_gain(model.a, model.b, np.diag(q_values), np.diag(r_values))
      controller_type = "lqr"
    except Exception as exc:
      reasons.append(f"DARE solve failed: {type(exc).__name__}: {exc}")

  if reasons:
    pd_values = np.asarray(pd_gain, dtype=np.float64)
    if pd_values.shape != (state_dim,):
      raise ValueError("pd_gain must provide one feedback gain for each state.")
    gain = pd_values.reshape(1, state_dim)
    controller_type = "pd"

  return ControllerDesign(
    controller_type=controller_type,
    gain=gain,
    model=model,
    controllability_rank=rank,
    heldout_nrmse=nrmse,
    fallback_reasons=tuple(reasons),
  )
