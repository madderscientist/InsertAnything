from __future__ import annotations

from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

from .base import InsertionShape, RoughTile, add_box

if TYPE_CHECKING:
    from ..geometry import Geometry


class LShape(InsertionShape):
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        r, c = radius, half_clearance
        p = r * np.array([[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]])
        peg = np.array([[-r+c, -r+c], [r-c, -r+c], [r-c, -c],
                        [-c, -c], [-c, r-c], [-r+c, r-c]])
        return p, peg, 0.0

    def add_plate_colliders(self, hole: ET.Element, assets: ET.Element, g: Geometry) -> None:
        r, outer = g.radius, g.plate_outer_radius_m or g.radius * 2
        bounds = (
            ("wall_left", [-outer, -outer], [-r, outer]),
            ("wall_right", [r, -outer], [outer, outer]),
            ("wall_bottom", [-r, -outer], [r, -r]),
            ("wall_top", [-r, r], [r, outer]),
            ("wall_notch", [0, 0], [r, r]),
        )
        for name, low, high in bounds:
            add_box(hole, name, np.array(low), np.array(high), -g.depth, 0, "0.5 0.5 0.6 1")

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        r, c = g.radius, g.clearance / 2
        add_box(body, "peg_bottom", np.array([-r+c, -r+c]), np.array([r-c, -c]),
                tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1")
        add_box(body, "peg_left", np.array([-r+c, -c]), np.array([-c, r-c]),
                tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1")

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
