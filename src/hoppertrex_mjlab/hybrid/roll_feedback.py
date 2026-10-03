"""Bounded roll feedback mapped onto identified left/right leg lift bases.

The local R0c leg-load sensitivity probe measured a monotone static response
across ``a = [-4, +4] mrad`` using
``left += a * LEFT_LIFT_BASIS`` and
``right -= a * RIGHT_LIFT_BASIS``.  The fitted response was
``d(roll) / d(a) = -0.6490 rad/rad`` (R^2 = 0.9983), so a positive feedback
amplitude counteracts a positive measured roll.  The hard 4 mrad cap prevents
runtime extrapolation beyond that measured bracket; development probes should
start at the narrower 1 mrad limit.

Feedback gains and authority default to zero in both simulation and portable
deployment paths.  Thus this module grants no authority unless a caller opts
in explicitly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .stair_dynamic import LEFT_LIFT_BASIS, RIGHT_LIFT_BASIS

ROLL_FEEDBACK_IDENTIFIED_SLOPE_RAD_PER_RAD = -0.649046170
ROLL_FEEDBACK_IDENTIFIED_R2 = 0.99833862
ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD = 0.004
ROLL_FEEDBACK_INITIAL_LIMIT_RAD = 0.001


def validate_roll_feedback_parameters(
  *,
  kp: float,
  kd: float,
  max_amplitude_rad: float,
) -> tuple[float, float, float]:
  """Validate bounded negative-feedback parameters.

  ``kp`` maps roll angle to lift amplitude and ``kd`` maps body roll rate to
  lift amplitude.  Both use the sign established by the local sensitivity
  probe.  Zero authority is valid and is the default-inert configuration.
  """

  values = {
    "roll_feedback_kp": float(kp),
    "roll_feedback_kd": float(kd),
    "roll_feedback_max_amplitude_rad": float(max_amplitude_rad),
  }
  for name, value in values.items():
    if not math.isfinite(value) or value < 0.0:
      raise ValueError(f"{name} must be finite and non-negative.")
  if (
    values["roll_feedback_max_amplitude_rad"]
    > ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD
  ):
    raise ValueError(
      "roll_feedback_max_amplitude_rad exceeds the locally identified "
      f"{ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD:g} rad bracket."
    )
  return (
    values["roll_feedback_kp"],
    values["roll_feedback_kd"],
    values["roll_feedback_max_amplitude_rad"],
  )


def roll_feedback_amplitude(
  roll_rad: Any,
  roll_rate_radps: Any,
  *,
  kp: float,
  kd: float,
  max_amplitude_rad: float,
) -> NDArray[np.float32] | np.float32:
  """Return ``clip(kp * roll + kd * roll_rate, +/- authority)`` in float32."""

  kp, kd, limit = validate_roll_feedback_parameters(
    kp=kp,
    kd=kd,
    max_amplitude_rad=max_amplitude_rad,
  )
  roll, roll_rate = np.broadcast_arrays(
    np.asarray(roll_rad, dtype=np.float32),
    np.asarray(roll_rate_radps, dtype=np.float32),
  )
  if not np.all(np.isfinite(roll)) or not np.all(np.isfinite(roll_rate)):
    raise ValueError("Roll angle and roll rate must be finite.")
  raw = np.float32(kp) * roll + np.float32(kd) * roll_rate
  return np.clip(raw, -np.float32(limit), np.float32(limit)).astype(np.float32)


def roll_feedback_leg_offsets(
  amplitude_rad: Any,
) -> NDArray[np.float32]:
  """Map scalar/batched amplitudes to the four runtime leg-joint offsets.

  Runtime joint order is ``(left thigh, right thigh, left knee, right knee)``.
  The right side receives the negative of its lift basis, matching the
  sensitivity experiment's differential composition.
  """

  amplitude = np.asarray(amplitude_rad, dtype=np.float32)
  if not np.all(np.isfinite(amplitude)):
    raise ValueError("Roll-feedback amplitude must be finite.")
  if np.any(
    np.abs(amplitude)
    > np.float32(ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD)
  ):
    raise ValueError("Roll-feedback amplitude exceeds the identified bracket.")
  basis = np.asarray(
    (
      LEFT_LIFT_BASIS[0],
      -RIGHT_LIFT_BASIS[0],
      LEFT_LIFT_BASIS[1],
      -RIGHT_LIFT_BASIS[1],
    ),
    dtype=np.float32,
  )
  return amplitude[..., None] * basis


__all__ = [
  "LEFT_LIFT_BASIS",
  "RIGHT_LIFT_BASIS",
  "ROLL_FEEDBACK_IDENTIFIED_R2",
  "ROLL_FEEDBACK_IDENTIFIED_SLOPE_RAD_PER_RAD",
  "ROLL_FEEDBACK_INITIAL_LIMIT_RAD",
  "ROLL_FEEDBACK_MAX_IDENTIFIED_AMPLITUDE_RAD",
  "roll_feedback_amplitude",
  "roll_feedback_leg_offsets",
  "validate_roll_feedback_parameters",
]
