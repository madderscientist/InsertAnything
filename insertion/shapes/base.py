"""插头与孔板共用的几何接口和碰撞体构造函数。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
import torch

if TYPE_CHECKING:
    from ..geometry import Geometry


@dataclass(frozen=True)
class RoughTile:
    cx: float
    cy: float
    sx: float
    sy: float
    sz: float
    yaw: float
    guards: tuple[tuple[float, float, float], ...]


def points_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Inclusive ray crossing, including concave polygons and their boundary."""
    start, end = polygon, torch.roll(polygon, -1, 0)
    edge = end-start
    relative = points[:, None, :]-start
    cross = edge[None, :, 0]*relative[..., 1]-edge[None, :, 1]*relative[..., 0]
    along = (relative*edge).sum(-1)
    boundary = ((cross.abs() <= 1e-10) & (along >= 0)
                & (along <= edge.square().sum(-1))).any(-1)
    crosses = (start[:, 1] > points[:, None, 1]) != (end[:, 1] > points[:, None, 1])
    dy = edge[:, 1]
    safe_dy = torch.where(dy != 0, dy, torch.ones_like(dy))
    intersection = start[:, 0] + (points[:, None, 1]-start[:, 1])*edge[:, 0]/safe_dy
    inside = ((crosses & (points[:, None, 0] < intersection)).sum(-1) % 2) == 1
    return inside | boundary


def inset_polygon(p: np.ndarray, distance: float) -> np.ndarray:
    # 将多边形每条边沿内法向平移固定距离，通过相邻半平面交点求新顶点
    edges = np.roll(p, -1, axis=0) - p
    inward = np.column_stack((-edges[:, 1], edges[:, 0]))
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    bounds = (inward * p).sum(1) + distance
    return np.array([np.linalg.solve(np.stack((inward[i - 1], inward[i])),
                                    np.array((bounds[i - 1], bounds[i])))
                     for i in range(len(p))])


def regular_polygon(radius: float, sides: int) -> np.ndarray:
    angles = np.arange(sides) * 2 * math.pi / sides + math.pi / 2
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def add_box(parent: ET.Element, name: str, low: np.ndarray, high: np.ndarray,
            z_low: float, z_high: float, color: str, yaw: float = 0.0) -> None:
    center = np.r_[(low + high) / 2, (z_low + z_high) / 2]
    size = np.r_[(high - low) / 2, (z_high - z_low) / 2]
    x, y = center[:2]
    center[:2] = (math.cos(yaw)*x - math.sin(yaw)*y,
                  math.sin(yaw)*x + math.cos(yaw)*y)
    ET.SubElement(parent, "geom", name=name, type="box",
                  pos=" ".join(map(str, center)), size=" ".join(map(str, size)),
                  quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
                  rgba=color)


def add_prism(parent: ET.Element, assets: ET.Element, name: str, polygon: np.ndarray,
              z_low: float, z_high: float, color: str) -> None:
    vertices = np.concatenate((np.column_stack((polygon, np.full(len(polygon), z_low))),
                               np.column_stack((polygon, np.full(len(polygon), z_high)))))
    ET.SubElement(assets, "mesh", name=name, vertex=" ".join(map(str, vertices.ravel())))
    ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name, rgba=color)


@dataclass(frozen=True)
class InsertionShape(ABC):
    """一对配合的插头和孔板；镜像轴为孔坐标系中穿过原点的 XY 方向向量。"""

    name: str
    rotation_order: int  # 0 表示忽略绕 Z 的对齐误差（圆形）
    mirror_axis_xy: tuple[float, float]
    train_by_default: bool = True

    @abstractmethod
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the hole polygon, peg polygon, and circle approximation error."""

    def add_plate_colliders(self, hole: ET.Element, assets: ET.Element, g: Geometry) -> None:
        # 拆成独立凸棱柱，避免整体凸包将中心孔封死
        p, out = g.hole_polygon, g.hole_polygon * 2
        if g.plate_outer_radius_m:
            edges = np.roll(p, -1, axis=0)-p
            inradius = np.min(np.abs(edges[:, 0]*p[:, 1]-edges[:, 1]*p[:, 0])
                              / np.linalg.norm(edges, axis=1))
            out = p*(g.plate_outer_radius_m/inradius)
        for i in range(len(p)):
            j = (i + 1) % len(p)
            add_prism(hole, assets, f"wall_{i}", np.array([p[i], out[i], out[j], p[j]]),
                      -g.depth, 0, "0.5 0.5 0.6 1")

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        add_prism(body, assets, "peg", g.peg_polygon, tip_z, tip_z + g.peg_length,
                  "0.8 0.6 0.1 1")

    def rough_ring_polygon(self, g: Geometry) -> np.ndarray:
        return g.hole_polygon

    def rough_infill_tiles(self, g: Geometry, *, overlap: float, half_height: float,
                           margin: float) -> tuple[RoughTile, ...]:
        return ()

    def contact_limits(self, rough_surface: bool) -> tuple[int, int]:
        """Return the MuJoCo Warp contact and constraint capacities."""
        # 粗糙面实测峰值：每世界约束数 ≤304、接触池利用率 <1%；溢出会被检测并终止该环境
        return (128, 640) if rough_surface else (64, 256)

    def peg_reach_radius(self, g: Geometry, peg_polygon: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(peg_polygon, dim=-1).amax())

    def minimum_peg_projection(self, g: Geometry, direction_xy: torch.Tensor,
                               peg_polygon: torch.Tensor) -> torch.Tensor:
        return (direction_xy @ peg_polygon.T).amin(dim=-1)

    def opening_contains(self, g: Geometry, points_xy: torch.Tensor,
                         hole_polygon: torch.Tensor) -> torch.Tensor:
        return points_in_polygon(points_xy, hole_polygon)


class RegularPolygonShape(InsertionShape):
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        p = regular_polygon(radius, self.rotation_order)
        return p, inset_polygon(p, half_clearance), 0.0
