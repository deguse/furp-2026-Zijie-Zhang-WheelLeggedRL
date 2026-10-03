"""Development-only ellipsoid collision proxy for the narrow drive wheels.

The source wheel collision geoms are cylinders with local-axis size
``(radius, half_width, 0)``. MJWarp routes cylinder-box through its generic
convex path, whose missing cylinder multicontact is the R0c failure under
investigation. The single preregistered proxy changes each collision geom to
an ellipsoid with local radii ``(radius, radius, half_width)``. It therefore
preserves the wheel's longitudinal/vertical radius, axial half-width, pose,
visual mesh, rigid-body inertia, and every controller/actuator parameter.

Nothing imports this module from a formal task configuration. Development
probes must opt in explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from hoppertrex_mjlab.assets.HopperTrex_CFG import get_spec

WHEEL_COLLISION_GEOM_NAMES = (
    "wheel_left_collision",
    "wheel_right_collision",
)


def make_ellipsoid_wheel_spec() -> mujoco.MjSpec:
    """Return the robot spec with only the two collision shapes replaced."""

    spec = get_spec()
    observed = []
    for name in WHEEL_COLLISION_GEOM_NAMES:
        geom = spec.geom(name)
        if geom is None or geom.type != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError(f"Expected cylinder collision geom {name!r}.")
        size = np.asarray(geom.size, dtype=np.float64).copy()
        if (
            size.shape != (3,)
            or not np.isfinite(size).all()
            or size[0] <= 0.0
            or size[1] <= 0.0
            or size[2] != 0.0
        ):
            raise ValueError(f"Wheel collision size for {name!r} is invalid: {size}.")
        observed.append(size[:2])
        radius, half_width = float(size[0]), float(size[1])
        geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        geom.size = np.asarray((radius, radius, half_width), dtype=np.float64)
    if not np.array_equal(observed[0], observed[1]):
        raise ValueError("Left/right source wheel collision sizes differ.")
    return spec


def ellipsoid_wheel_spec_fn() -> Callable[[], mujoco.MjSpec]:
    """Return a named zero-argument factory suitable for ``EntityCfg.spec_fn``."""

    return make_ellipsoid_wheel_spec


__all__ = [
    "WHEEL_COLLISION_GEOM_NAMES",
    "ellipsoid_wheel_spec_fn",
    "make_ellipsoid_wheel_spec",
]
