from __future__ import annotations

import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .base import InsertionShape, inset_polygon, regular_polygon

if TYPE_CHECKING:
    from ..geometry import Geometry


def validate_circle_sides(sides: int | None) -> None:
    if sides is not None and (type(sides) is not int or sides < 3):
        raise ValueError("circle_sides must be an integer >= 3 or None")


class CircleShape(InsertionShape):
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        # 控制圆形多边形孔壁的弦高误差不超过单侧间隙的 5%
        validate_circle_sides(circle_sides)
        n = (circle_sides if circle_sides is not None else
             max(128, math.ceil(math.pi / math.acos(1 - half_clearance * 0.05 / radius))))
        error = radius * (1 - math.cos(math.pi / n))
        if error >= half_clearance:
            raise ValueError("circle polygon approximation closes the radial clearance")
        p = regular_polygon(radius, n)
        return p, inset_polygon(p, half_clearance), error

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        ET.SubElement(body, "geom", name="peg", type="cylinder",
                      pos=f"0 0 {tip_z + g.peg_length / 2}",
                      size=f"{g.radius-g.clearance/2} {g.peg_length/2}", rgba="0.8 0.6 0.1 1")

    def rough_ring_polygon(self, g: Geometry) -> np.ndarray:
        n = 12
        angles = np.arange(n) * 2 * math.pi / n
        return (g.radius / math.cos(math.pi / n)) * np.column_stack((np.cos(angles), np.sin(angles)))

    def contact_limits(self, rough_surface: bool) -> tuple[int, int]:
        return 512, 2048

    def peg_reach_radius(self, g: Geometry, peg_polygon: torch.Tensor) -> float:
        return g.radius

    def minimum_peg_projection(self, g: Geometry, direction_xy: torch.Tensor,
                               peg_polygon: torch.Tensor) -> torch.Tensor:
        return -(g.radius-g.clearance/2) * torch.linalg.vector_norm(direction_xy, dim=-1)

    def opening_contains(self, g: Geometry, points_xy: torch.Tensor,
                         hole_polygon: torch.Tensor) -> torch.Tensor:
        return points_xy.square().sum(-1) <= g.radius**2


SHAPE = CircleShape("circle", rotation_order=0, mirror_axis_xy=(0.0, 1.0), train_by_default=False)
