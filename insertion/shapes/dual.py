"""双脚扁插头与三槽插面：中间槽与两侧目标槽同尺寸同深度，单脚误入中间槽时插头无法到底"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..transforms import rotate
from .base import BoxShape, PreparedGeometry, rectangle_corners

if TYPE_CHECKING:
    from ..geometry import Geometry

# 国标两脚扁插（radius=10 mm 时的米）：插脚厚、宽，槽孔间距为插脚间距 12.7 mm 的一半；随机化 ±20%
NOMINAL_BLADE = (0.0015, 0.0064, 0.00635)
BLADE_SPREAD = 0.2


def _xy(*values: torch.Tensor) -> torch.Tensor:
    return torch.stack(torch.broadcast_tensors(*values), -1)


class DualBladeShape(BoxShape):
    """params = (插脚厚 t, 插脚宽 w, 槽孔间距 p)；两片插脚位于插头尖端两侧 ±p"""

    def nominal_params(self, g: Geometry) -> tuple[float, ...]:
        return tuple(x * g.radius / 0.01 for x in NOMINAL_BLADE)

    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        # 名义几何只用于粗糙面与可视化：孔口取三槽外包矩形，插头取两脚外包矩形
        t, w, p = (x * radius / 0.01 for x in NOMINAL_BLADE)
        unit = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        return (unit*(p + t/2 + half_clearance, w/2 + half_clearance), unit*(p + t/2, w/2), 0.0)

    def sample_params(self, g: Geometry, n: int, uniform) -> torch.Tensor:
        factor = uniform(n, 3, 1 - BLADE_SPREAD, 1 + BLADE_SPREAD)
        for i, value in enumerate(self.nominal_params(g)):
            factor[:, i] *= value
        return factor

    def _slots(self, g: Geometry, params: torch.Tensor):
        c = params[:, 3] / 2
        return params[:, 0]/2 + c, params[:, 1]/2 + c, params[:, 2]

    def plate_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        tx, wy, p = self._slots(g, params)
        o, z = g.plate_outer_radius_m or g.radius*2, params[:, 0]*0
        return {
            "wall_top": (_xy(z-o, wy), _xy(z+o, z+o)),
            "wall_bottom": (_xy(z-o, z-o), _xy(z+o, -wy)),
            "wall_c0": (_xy(z-o, -wy), _xy(-p-tx, wy)),
            "wall_c1": (_xy(-p+tx, -wy), _xy(-tx, wy)),
            "wall_c2": (_xy(tx, -wy), _xy(p-tx, wy)),
            "wall_c3": (_xy(p+tx, -wy), _xy(z+o, wy)),
        }

    def peg_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        t, w, p = params[:, 0]/2, params[:, 1]/2, params[:, 2]
        return {"peg_blade_neg": (_xy(-p-t, -w), _xy(-p+t, w)),
                "peg_blade_pos": (_xy(p-t, -w), _xy(p+t, w))}

    def hole_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        tx, wy, p = self._slots(g, params)
        return rectangle_corners(p + tx, wy)

    def peg_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        t, w, p = params[:, 0]/2, params[:, 1]/2, params[:, 2]
        return torch.cat((rectangle_corners(t, w, -p), rectangle_corners(t, w, p)), 1)

    def peg_vertices(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        return self.peg_outline(geo.g, params)

    def _in_slot(self, geo: PreparedGeometry, params: torch.Tensor, points_xy: torch.Tensor,
                 slot: int | None) -> torch.Tensor:
        tx, wy, p = (x[:, None] for x in self._slots(geo.g, params))
        inside_y = points_xy[..., 1].abs() <= wy
        if slot is not None:
            return inside_y & ((points_xy[..., 0] - slot*p).abs() <= tx)
        nearest = torch.minimum((points_xy[..., 0] - p).abs(), (points_xy[..., 0] + p).abs())
        return inside_y & (torch.minimum(nearest, points_xy[..., 0].abs()) <= tx)

    def opening_contains(self, geo: PreparedGeometry, params: torch.Tensor,
                         points_xy: torch.Tensor) -> torch.Tensor:
        return self._in_slot(geo, params, points_xy, None)

    def inserted(self, geo: PreparedGeometry, params: torch.Tensor, tip: torch.Tensor,
                 q: torch.Tensor) -> torch.Tensor:
        # 两片插脚尖端须分别位于两侧目标槽内；插头旋转 180° 后两脚互换同样成立
        offset = torch.stack((params[:, 2], torch.zeros_like(params[:, 2]), torch.zeros_like(params[:, 2])), -1)
        blades = torch.stack((tip + rotate(q, offset), tip - rotate(q, offset)), 1)[..., :2]
        pos, neg = self._in_slot(geo, params, blades, 1), self._in_slot(geo, params, blades, -1)
        return (pos[:, 0] & neg[:, 1]) | (neg[:, 0] & pos[:, 1])

    def opening_radius(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        tx, wy, p = self._slots(geo.g, params)
        return torch.hypot(p + tx, wy)

    def rim_bounds(self, g: Geometry, params: torch.Tensor,
                   width: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # 抬高区域为三槽外包矩形外扩 w，槽间的两块隔墙一并抬高
        tx, wy, p = self._slots(g, params)
        x, w = p + tx, width
        return [
            (_xy(-x-w, -wy-w), _xy(-x, wy+w)),
            (_xy(x, -wy-w), _xy(x+w, wy+w)),
            (_xy(-x, -wy-w), _xy(x, -wy)),
            (_xy(-x, wy), _xy(x, wy+w)),
            (_xy(-p+tx, -wy), _xy(-tx, wy)),
            (_xy(tx, -wy), _xy(p-tx, wy)),
        ]

    def contact_limits(self, rough_surface: bool) -> tuple[int, int]:
        return (128, 640)


SHAPE = DualBladeShape("dual", rotation_order=2, mirror_axis_xy=(0.0, 1.0), train_by_default=False)
