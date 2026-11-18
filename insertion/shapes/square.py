from __future__ import annotations

import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

from .base import RegularPolygonShape, add_box

if TYPE_CHECKING:
    from ..geometry import Geometry


class SquareShape(RegularPolygonShape):
    def add_plate_colliders(self, hole: ET.Element, assets: ET.Element, g: Geometry) -> None:
        a = g.radius / math.sqrt(2)
        b = g.plate_outer_radius_m or 2*a
        # 用不重叠的盒体拼接方形板，保持内壁间隙
        bounds = (
            ([-b, -b], [-a, b]),
            ([a, -b], [b, b]),
            ([-a, -b], [a, -a]),
            ([-a, a], [a, b]),
        )
        for i, (low, high) in enumerate(bounds):
            add_box(hole, f"wall_{i}", np.array(low), np.array(high),
                    -g.depth, 0, "0.5 0.5 0.6 1", yaw=math.pi/4)

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        a = g.radius / math.sqrt(2) - g.clearance / 2
        add_box(body, "peg", np.array([-a, -a]), np.array([a, a]),
                tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1", yaw=math.pi/4)


SHAPE = SquareShape("square", rotation_order=4, mirror_axis_xy=(0.0, 1.0))
