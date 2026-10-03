"""Pure CAN payload codecs for HopperTrex motors.

No transport is implemented here.  Callers must provide the configured DaMiao
mapping ranges and a calibrated RMD output torque constant explicitly.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

DM_KP_MIN = 0.0
DM_KP_MAX = 500.0
DM_KD_MIN = 0.0
DM_KD_MAX = 5.0
RMD_TORQUE_COMMAND = 0xA1
RMD_STATUS_2_COMMAND = 0x9C
RMD_IQ_RESOLUTION_A = 0.01


@dataclass(frozen=True)
class DmMitMapping:
  position_max_rad: float
  velocity_max_rad_s: float
  torque_max_nm: float

  def __post_init__(self) -> None:
    values = (
      self.position_max_rad,
      self.velocity_max_rad_s,
      self.torque_max_nm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
      raise ValueError("DaMiao MIT mapping limits must be finite and positive.")


@dataclass(frozen=True)
class DmMitFeedback:
  status_code: int
  motor_id: int
  position_rad: float
  velocity_rad_s: float
  torque_estimate_nm: float
  mos_temperature_c: float
  rotor_temperature_c: float


@dataclass(frozen=True)
class RmdTorqueFeedback:
  command: int
  temperature_c: float
  iq_current_a: float
  output_speed_rad_s: float
  output_angle_rad: float


def _float_to_uint(value: float, lower: float, upper: float, bits: int) -> int:
  if not math.isfinite(value):
    raise ValueError("CAN command values must be finite.")
  if value < lower or value > upper:
    raise ValueError(f"Value {value} is outside [{lower}, {upper}].")
  return int((value - lower) * ((1 << bits) - 1) / (upper - lower))


def _uint_to_float(value: int, lower: float, upper: float, bits: int) -> float:
  if value < 0 or value > (1 << bits) - 1:
    raise ValueError(f"Unsigned {bits}-bit value is out of range.")
  return lower + value * (upper - lower) / ((1 << bits) - 1)


def encode_dm_mit_command(
  *,
  position_rad: float,
  velocity_rad_s: float,
  kp: float,
  kd: float,
  feedforward_torque_nm: float,
  mapping: DmMitMapping,
) -> bytes:
  """Encode the standard 8-byte DaMiao MIT command payload."""

  p = _float_to_uint(
    position_rad,
    -mapping.position_max_rad,
    mapping.position_max_rad,
    16,
  )
  v = _float_to_uint(
    velocity_rad_s,
    -mapping.velocity_max_rad_s,
    mapping.velocity_max_rad_s,
    12,
  )
  kp_u = _float_to_uint(kp, DM_KP_MIN, DM_KP_MAX, 12)
  kd_u = _float_to_uint(kd, DM_KD_MIN, DM_KD_MAX, 12)
  torque = _float_to_uint(
    feedforward_torque_nm,
    -mapping.torque_max_nm,
    mapping.torque_max_nm,
    12,
  )
  return bytes(
    (
      (p >> 8) & 0xFF,
      p & 0xFF,
      (v >> 4) & 0xFF,
      ((v & 0xF) << 4) | ((kp_u >> 8) & 0xF),
      kp_u & 0xFF,
      (kd_u >> 4) & 0xFF,
      ((kd_u & 0xF) << 4) | ((torque >> 8) & 0xF),
      torque & 0xFF,
    )
  )


def decode_dm_mit_feedback(data: bytes, *, mapping: DmMitMapping) -> DmMitFeedback:
  """Decode DaMiao status/position/velocity/torque/temperature feedback."""

  if len(data) != 8:
    raise ValueError("DaMiao feedback frame must contain exactly 8 bytes.")
  position = (data[1] << 8) | data[2]
  velocity = (data[3] << 4) | (data[4] >> 4)
  torque = ((data[4] & 0xF) << 8) | data[5]
  return DmMitFeedback(
    status_code=data[0] >> 4,
    motor_id=data[0] & 0xF,
    position_rad=_uint_to_float(
      position,
      -mapping.position_max_rad,
      mapping.position_max_rad,
      16,
    ),
    velocity_rad_s=_uint_to_float(
      velocity,
      -mapping.velocity_max_rad_s,
      mapping.velocity_max_rad_s,
      12,
    ),
    torque_estimate_nm=_uint_to_float(
      torque,
      -mapping.torque_max_nm,
      mapping.torque_max_nm,
      12,
    ),
    mos_temperature_c=float(data[6]),
    rotor_temperature_c=float(data[7]),
  )


def encode_rmd_torque_current_command(iq_current_a: float) -> bytes:
  """Encode RMD 0xA1 iq target (signed int16, 0.01 A/LSB)."""

  if not math.isfinite(iq_current_a):
    raise ValueError("RMD iq target must be finite.")
  raw = round(iq_current_a / RMD_IQ_RESOLUTION_A)
  if not -32768 <= raw <= 32767:
    raise ValueError("RMD iq target exceeds the int16 protocol range.")
  low, high = struct.pack("<h", raw)
  return bytes((RMD_TORQUE_COMMAND, 0, 0, 0, low, high, 0, 0))


def decode_rmd_torque_feedback(data: bytes) -> RmdTorqueFeedback:
  """Decode an RMD 0xA1 or status-2 reply in documented output units."""

  if len(data) != 8:
    raise ValueError("RMD feedback frame must contain exactly 8 bytes.")
  if data[0] not in (RMD_TORQUE_COMMAND, RMD_STATUS_2_COMMAND):
    raise ValueError("RMD frame is neither a 0xA1 nor 0x9C feedback reply.")
  temperature = struct.unpack("<b", data[1:2])[0]
  iq = struct.unpack("<h", data[2:4])[0] * RMD_IQ_RESOLUTION_A
  speed_degrees_s = struct.unpack("<h", data[4:6])[0]
  angle_degrees = struct.unpack("<h", data[6:8])[0]
  return RmdTorqueFeedback(
    command=data[0],
    temperature_c=float(temperature),
    iq_current_a=float(iq),
    output_speed_rad_s=math.radians(speed_degrees_s),
    output_angle_rad=math.radians(angle_degrees),
  )


def wheel_torque_to_iq_current(
  torque_nm: float,
  *,
  torque_constant_nm_per_a: float,
) -> float:
  """Convert output torque to iq using an explicitly calibrated constant."""

  values = (torque_nm, torque_constant_nm_per_a)
  if any(not math.isfinite(value) for value in values):
    raise ValueError("Torque and torque constant must be finite.")
  if torque_constant_nm_per_a <= 0.0:
    raise ValueError("Torque constant must be positive.")
  return torque_nm / torque_constant_nm_per_a


def iq_current_to_wheel_torque(
  iq_current_a: float,
  *,
  torque_constant_nm_per_a: float,
) -> float:
  """Convert measured iq to an output-torque estimate after calibration."""

  values = (iq_current_a, torque_constant_nm_per_a)
  if any(not math.isfinite(value) for value in values):
    raise ValueError("Current and torque constant must be finite.")
  if torque_constant_nm_per_a <= 0.0:
    raise ValueError("Torque constant must be positive.")
  return iq_current_a * torque_constant_nm_per_a

