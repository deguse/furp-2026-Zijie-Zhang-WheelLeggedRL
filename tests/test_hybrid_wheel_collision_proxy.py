from __future__ import annotations

import unittest

import mujoco
import numpy as np

from hoppertrex_mjlab.assets.HopperTrex_CFG import get_spec
from hoppertrex_mjlab.hybrid.wheel_collision_proxy import (
    WHEEL_COLLISION_GEOM_NAMES,
    make_ellipsoid_wheel_spec,
)


class WheelCollisionProxyTest(unittest.TestCase):
    def test_proxy_changes_only_type_and_axis_preserving_size(self):
        baseline = get_spec()
        proxy = make_ellipsoid_wheel_spec()
        for name in WHEEL_COLLISION_GEOM_NAMES:
            source_geom = baseline.geom(name)
            proxy_geom = proxy.geom(name)
            self.assertEqual(source_geom.type, mujoco.mjtGeom.mjGEOM_CYLINDER)
            self.assertEqual(proxy_geom.type, mujoco.mjtGeom.mjGEOM_ELLIPSOID)
            radius, half_width = source_geom.size[:2]
            np.testing.assert_array_equal(
                proxy_geom.size,
                np.asarray((radius, radius, half_width)),
            )
            np.testing.assert_array_equal(proxy_geom.pos, source_geom.pos)
            np.testing.assert_array_equal(proxy_geom.quat, source_geom.quat)

    def test_proxy_preserves_compiled_rigid_body_inertia(self):
        baseline = get_spec().compile()
        proxy = make_ellipsoid_wheel_spec().compile()
        for name in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "jnt_pos",
            "jnt_axis",
        ):
            np.testing.assert_array_equal(getattr(proxy, name), getattr(baseline, name))


if __name__ == "__main__":
    unittest.main()
