from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import BoxShape, RoughTile, polygon_from_xy

if TYPE_CHECKING:
    from ..geometry import Geometry

L_SCALE_RANGE = (0.6, 1.0)


def _xy(*values: torch.Tensor) -> torch.Tensor:
    return torch.stack(torch.broadcast_tensors(*values), -1)


class LShape(BoxShape):
    """params = (外包方形半边长缩放 s, 0, 0)"""

    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        r, c = radius, half_clearance
        p = r * np.array([[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]])
        peg = np.array([[-r+c, -r+c], [r-c, -r+c], [r-c, -c],
                        [-c, -c], [-c, r-c], [-r+c, r-c]])
        return p, peg, 0.0

    def sample_params(self, g: Geometry, n: int, uniform) -> torch.Tensor:
        scale = uniform(n, 1, *L_SCALE_RANGE)
        return torch.cat((scale, torch.zeros_like(scale), torch.zeros_like(scale)), -1)

    def plate_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        r, o, z = g.radius*params[:, 0], g.plate_outer_radius_m or g.radius*2, params[:, 0]*0
        return {
            "wall_left": (_xy(z-o, z-o), _xy(-r, z+o)),
            "wall_right": (_xy(r, z-o), _xy(z+o, z+o)),
            "wall_bottom": (_xy(-r, z-o), _xy(r, -r)),
            "wall_top": (_xy(-r, r), _xy(r, z+o)),
            "wall_notch": (_xy(z, z), _xy(r, r)),
        }

    def peg_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        r, c = g.radius*params[:, 0], g.clearance/2
        return {
            "peg_bottom": (_xy(-r+c, -r+c), _xy(r-c, -r*0-c)),
            "peg_left": (_xy(-r+c, -r*0-c), _xy(-r*0-c, r-c)),
        }

    def hole_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        r = g.radius*params[:, 0]
        z = r*0
        return polygon_from_xy((-r, -r), (r, -r), (r, z), (z, z), (z, r), (-r, r))

    def peg_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        r, c = g.radius*params[:, 0], g.clearance/2
        z = r*0 - c
        return polygon_from_xy((-r+c, -r+c), (r-c, -r+c), (r-c, z), (z, z), (z, r-c), (-r+c, r-c))

    def opening_radius(self, geo, params: torch.Tensor) -> torch.Tensor:
        return geo.g.radius*params[:, 0]

    def rim_bounds(self, g: Geometry, params: torch.Tensor,
                   width: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # 凹角处的盒体不越过孔口，其余外角互相搭接
        r, w, z = g.radius*params[:, 0], width, params[:, 0]*0
        return [
            (_xy(-r-w, -r-w), _xy(-r, r+w)),
            (_xy(-r, -r-w), _xy(r+w, -r)),
            (_xy(r, -r), _xy(r+w, w)),
            (_xy(z, z), _xy(r+w, w)),
            (_xy(z, z), _xy(w, r+w)),
            (_xy(-r-w, r), _xy(w, r+w)),
        ]

    def rough_ring_polygon(self, g: Geometry) -> np.ndarray:
        return g.radius * np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])

    def rough_infill_tiles(self, g: Geometry, *, overlap: float, half_height: float,
                           margin: float) -> tuple[RoughTile, ...]:
        tiles: list[RoughTile] = []
        n_notch = 3
        for ix in range(n_notch):
            for iy in range(n_notch):
                x0 = margin + ix * (g.radius - margin) / n_notch
                x1 = margin + (ix + 1) * (g.radius - margin) / n_notch
                y0 = margin + iy * (g.radius - margin) / n_notch
                y1 = margin + (iy + 1) * (g.radius - margin) / n_notch
                xb0 = margin if ix == 0 else x0 - overlap
                xb1 = x1 + overlap
                yb0 = margin if iy == 0 else y0 - overlap
                yb1 = y1 + overlap
                tiles.append(RoughTile(
                    0.5 * (xb0 + xb1), 0.5 * (yb1 + yb0),
                    0.5 * (xb1 - xb0), 0.5 * (yb1 - yb0), half_height, 0.0,
                    ((1.0, 0.0, margin), (0.0, 1.0, margin)),
                ))
        return tuple(tiles)


SHAPE = LShape("l", rotation_order=1, mirror_axis_xy=(1.0, 1.0))
