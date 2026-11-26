"""逐环境随机长宽的矩形插头与矩形孔"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import BoxShape, rectangle_corners

if TYPE_CHECKING:
    from ..geometry import Geometry

# 孔口半长、半宽范围（单位：radius=10 mm 时的米）：下限为 Type-C 母座 8.34 x 2.56 mm，上限为原方孔边长 14.14 mm
HALF_LONG_RANGE = (0.00417, 0.00707)
HALF_SHORT_RANGE = (0.00128, 0.00707)


def _xy(*values: torch.Tensor) -> torch.Tensor:
    return torch.stack(torch.broadcast_tensors(*values), -1)


class RectangleShape(BoxShape):
    """params = (孔口 X 半长 a, 孔口 Y 半宽 b, 0)"""

    def nominal_params(self, g: Geometry) -> tuple[float, ...]:
        scale = g.radius / 0.01
        return (HALF_LONG_RANGE[1]*scale, HALF_LONG_RANGE[1]*scale/2, 0.0)

    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        a, b, c = HALF_LONG_RANGE[1]*radius/0.01, HALF_LONG_RANGE[1]*radius/0.02, half_clearance
        unit = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        return unit*(a, b), unit*(a-c, b-c), 0.0

    def sample_params(self, g: Geometry, n: int, uniform) -> torch.Tensor:
        scale = g.radius / 0.01
        a = uniform(n, 1, HALF_LONG_RANGE[0]*scale, HALF_LONG_RANGE[1]*scale)
        b = uniform(n, 1, HALF_SHORT_RANGE[0]*scale, HALF_SHORT_RANGE[1]*scale)
        return torch.cat((a, b, torch.zeros_like(a)), -1)

    def plate_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        a, b, o = params[:, 0], params[:, 1], g.plate_outer_radius_m or g.radius*2
        return {
            "wall_left": (_xy(-a*0-o, -b*0-o), _xy(-a, b*0+o)),
            "wall_right": (_xy(a, -b*0-o), _xy(a*0+o, b*0+o)),
            "wall_bottom": (_xy(-a, -b*0-o), _xy(a, -b)),
            "wall_top": (_xy(-a, b), _xy(a, b*0+o)),
        }

    def peg_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        a, b, c = params[:, 0], params[:, 1], g.clearance/2
        return {"peg": (_xy(-a+c, -b+c), _xy(a-c, b-c))}

    def hole_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        return rectangle_corners(params[:, 0], params[:, 1])

    def peg_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        c = g.clearance/2
        return rectangle_corners(params[:, 0] - c, params[:, 1] - c)

    def opening_radius(self, geo, params: torch.Tensor) -> torch.Tensor:
        return torch.hypot(params[:, 0], params[:, 1])

    def rim_bounds(self, g: Geometry, params: torch.Tensor,
                   width: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        a, b, w = params[:, 0], params[:, 1], width
        return [
            (_xy(-a-w, -b-w), _xy(-a, b+w)),
            (_xy(a, -b-w), _xy(a+w, b+w)),
            (_xy(-a, -b-w), _xy(a, -b)),
            (_xy(-a, b), _xy(a, b+w)),
        ]


SHAPE = RectangleShape("rectangle", rotation_order=2, mirror_axis_xy=(0.0, 1.0), train_by_default=False)
