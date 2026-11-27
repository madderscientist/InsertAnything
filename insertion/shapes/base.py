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


SHAPE_PARAM_DIM = 3
# 未启用的尺寸档位或围框碍撞体移到远处，不参与接触
PARKED_OFFSET_M = 1.0


def points_in_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Inclusive ray crossing for (N, 2) points against a shared (V, 2) or per-point (N, V, 2) polygon."""
    if polygon.dim() == 2:
        polygon = polygon.expand(points.shape[0], -1, -1)
    start, end = polygon, torch.roll(polygon, -1, 1)
    edge = end-start
    relative = points[:, None, :]-start
    cross = edge[..., 0]*relative[..., 1]-edge[..., 1]*relative[..., 0]
    along = (relative*edge).sum(-1)
    boundary = ((cross.abs() <= 1e-10) & (along >= 0)
                & (along <= edge.square().sum(-1))).any(-1)
    crosses = (start[..., 1] > points[:, None, 1]) != (end[..., 1] > points[:, None, 1])
    dy = edge[..., 1]
    safe_dy = torch.where(dy != 0, dy, torch.ones_like(dy))
    intersection = start[..., 0] + (points[:, None, 1]-start[..., 1])*edge[..., 0]/safe_dy
    inside = ((crosses & (points[:, None, 0] < intersection)).sum(-1) % 2) == 1
    return inside | boundary


def points_in_polygons(points: torch.Tensor, polygons: torch.Tensor) -> torch.Tensor:
    """(n, M, 2) points against per-environment (n, V, 2) polygons, returning (n, M)."""
    n, m = points.shape[:2]
    flat = polygons[:, None].expand(-1, m, -1, -1).reshape(n*m, *polygons.shape[1:])
    return points_in_polygon(points.reshape(n*m, 2), flat).reshape(n, m)


def box_from_bounds(low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (low + high) / 2, (high - low) / 2


def polygon_from_xy(*vertices: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """由逐环境坐标张量拼出 (n, V, 2) 多边形；不创建主机常量，可在 CUDA Graph 捕获中使用"""
    return torch.stack([torch.stack(torch.broadcast_tensors(x, y), -1) for x, y in vertices], 1)


def rectangle_corners(half_x: torch.Tensor, half_y: torch.Tensor,
                      center_x: torch.Tensor | None = None) -> torch.Tensor:
    low, high = -half_x, half_x
    if center_x is not None:
        low, high = low + center_x, high + center_x
    return polygon_from_xy((low, -half_y), (high, -half_y), (high, half_y), (low, half_y))


@dataclass(frozen=True)
class PreparedGeometry:
    """单个仿真组的几何常量；张量须在 CUDA Graph 捕获前创建"""
    g: Geometry
    hole: torch.Tensor
    peg: torch.Tensor


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
    """一对配合的插头和孔板；镜像轴为孔坐标系中穿过原点的 XY 方向向量。

    几何查询均按环境接收参数 params (n, SHAPE_PARAM_DIM+1)：前 SHAPE_PARAM_DIM 列为形状参数，
    末列为本回合直径间隙；默认实现对应固定名义几何。
    """

    name: str
    rotation_order: int  # 0 表示忽略绕 Z 的对齐误差（圆形）
    mirror_axis_xy: tuple[float, float]
    train_by_default: bool = True

    def prepare(self, g: Geometry, device: str) -> PreparedGeometry:
        return PreparedGeometry(g, torch.tensor(g.hole_polygon, dtype=torch.float32, device=device),
                                torch.tensor(g.peg_polygon, dtype=torch.float32, device=device))

    def nominal_params(self, g: Geometry) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)

    def sample_params(self, g: Geometry, n: int, uniform) -> torch.Tensor:
        """uniform(n, columns, low, high) 为仿真组的随机数发生函数"""
        params = uniform(n, SHAPE_PARAM_DIM, 0., 0.)
        for i, value in enumerate(self.nominal_params(g)):
            params[:, i] += value
        return params

    def apply_params(self, plant, ids: torch.Tensor, params: torch.Tensor) -> None:
        """把形状参数写入 plant.geom_base_pos 与物理模型的碍撞体尺寸"""

    def peg_vertices(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        return geo.peg.expand(params.shape[0], -1, -1)

    def minimum_peg_projection(self, geo: PreparedGeometry, params: torch.Tensor,
                               direction_xy: torch.Tensor) -> torch.Tensor:
        return (self.peg_vertices(geo, params) * direction_xy[:, None, :]).sum(-1).amin(-1)

    def opening_contains(self, geo: PreparedGeometry, params: torch.Tensor,
                         points_xy: torch.Tensor) -> torch.Tensor:
        """points_xy (n, M, 2) 是否位于任一孔口内，返回 (n, M)"""
        return points_in_polygons(points_xy, geo.hole.expand(params.shape[0], -1, -1))

    def inserted(self, geo: PreparedGeometry, params: torch.Tensor, tip: torch.Tensor,
                 q: torch.Tensor) -> torch.Tensor:
        """成功判定的水平条件：插头位于目标孔内"""
        return self.opening_contains(geo, params, tip[:, None, :2])[:, 0]

    def opening_radius(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        return params.new_full((params.shape[0],), geo.g.radius)

    def rim_yaws(self, g: Geometry) -> tuple[float, ...]:
        """孔口围框盒体的偏航角；空表示该形状不支持围框"""
        return ()

    def rim_boxes(self, geo: PreparedGeometry, params: torch.Tensor,
                  width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """围框盒体的中心 (n, B, 2) 与局部系半尺寸 (n, B, 2)，内壁与孔壁齐平"""
        raise NotImplementedError(f"{self.name} does not support a raised rim")

    def add_rim_colliders(self, hole: ET.Element, g: Geometry) -> None:
        for i, yaw in enumerate(self.rim_yaws(g)):
            ET.SubElement(hole, "geom", name=f"rim_{i}", type="box",
                          pos=f"0 0 {-PARKED_OFFSET_M}", size="0.001 0.001 0.001",
                          quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}", rgba="0.85 0.35 0.25 1")

    @abstractmethod
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the hole polygon, peg polygon, and circle approximation error."""

    def add_plate_colliders(self, hole: ET.Element, assets: ET.Element, g: Geometry) -> None:
        add_walls(hole, g, g.hole_polygon, "wall_")

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


def add_walls(hole: ET.Element, g: Geometry, p: np.ndarray, prefix: str) -> None:
    # 每条边一块盒体，取该边孔壁梯形在边坐标系下的包围盒；相邻盒体在角上重叠，沿法向外移后孔口仍封闭
    out = p * 2
    if g.plate_outer_radius_m:
        edges = np.roll(p, -1, axis=0)-p
        inradius = np.min(np.abs(edges[:, 0]*p[:, 1]-edges[:, 1]*p[:, 0])
                          / np.linalg.norm(edges, axis=1))
        out = p*(g.plate_outer_radius_m/inradius)
    for i in range(len(p)):
        j = (i + 1) % len(p)
        yaw = math.atan2(*(p[j]-p[i])[::-1])
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([p[i], out[i], out[j], p[j]]) @ np.array([[c, -s], [s, c]])
        add_box(hole, f"{prefix}{i}", local.min(0), local.max(0), -g.depth, 0, "0.5 0.5 0.6 1", yaw)


class RegularPolygonShape(InsertionShape):
    def cross_sections(self, radius: float, half_clearance: float,
                       circle_sides: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        p = regular_polygon(radius, self.rotation_order)
        return p, inset_polygon(p, half_clearance), 0.0


POLYGON_SCALES = (0.6, 0.8, 1.0)


class ScaledPolygonShape(RegularPolygonShape):
    """插头网格棱柱无法逐环境缩放：随机化时插头预置尺寸档，每个环境只启用一档；
    孔壁盒体按尺寸档与间隙逐环境沿法向平移

    params = (外接圆缩放 s, 档位 k, 0)
    """

    def nominal_params(self, g: Geometry) -> tuple[float, ...]:
        return (1.0, float(len(POLYGON_SCALES) - 1), 0.0)

    def sample_params(self, g: Geometry, n: int, uniform) -> torch.Tensor:
        k = (uniform(n, 1, 0., 1.) * len(POLYGON_SCALES)).floor().clamp_max(len(POLYGON_SCALES) - 1)
        scale = torch.zeros_like(k)
        for i, s in enumerate(POLYGON_SCALES):
            scale = torch.where(k == i, s, scale)
        return torch.cat((scale, k, torch.zeros_like(k)), -1)

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        if not g.randomized:
            return super().add_peg_colliders(body, assets, g, tip_z)
        for k, s in enumerate(POLYGON_SCALES):
            peg = inset_polygon(regular_polygon(g.radius*s, self.rotation_order), g.clearance/2)
            add_prism(body, assets, f"peg_{k}", peg, tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1")

    def apply_params(self, plant, ids: torch.Tensor, params: torch.Tensor) -> None:
        active = params[:, 1].long()
        for k in range(len(POLYGON_SCALES)):
            parked = (active != k).float()[:, None] * PARKED_OFFSET_M
            for gid in plant.geom_ids_with_prefix(f"peg_{k}"):
                offset = torch.cat((torch.zeros_like(parked), torch.zeros_like(parked), parked), -1)
                plant.geom_base_pos[ids, gid] = plant.default_geom_pos[gid] + offset
        # 外缘内切圆半径固定，盒体只改内壁位置与厚度
        g, half_angle = plant.geometry, math.pi/self.rotation_order
        inner = self._apothem(g, params)
        outer = g.plate_outer_radius_m or 2*g.radius*math.cos(half_angle)
        normal = self._normals(params)
        half = torch.stack((inner*0 + outer*math.tan(half_angle), (outer-inner)/2), -1)
        for i in range(self.rotation_order):
            plant.set_box(ids, f"wall_{i}", ((inner+outer)/2)[:, None]*normal[i], half)

    def _apothem(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        # 插头按建模间隙固定，回合间隙多出的部分由孔壁外移补足
        return g.radius*params[:, 0]*math.cos(math.pi/self.rotation_order) + (params[:, 3]-g.clearance)/2

    def _normals(self, params: torch.Tensor) -> torch.Tensor:
        half_angle = math.pi/self.rotation_order
        angle = torch.arange(self.rotation_order, device=params.device)*(2*half_angle) + math.pi/2 + half_angle
        return torch.stack((angle.cos(), angle.sin()), -1)

    def _polygon(self, radius: torch.Tensor) -> torch.Tensor:
        sides = self.rotation_order
        angles = torch.arange(sides, device=radius.device) * (2*math.pi/sides) + math.pi/2
        return radius[:, None, None] * torch.stack((angles.cos(), angles.sin()), -1)

    def peg_vertices(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        if not geo.g.randomized:
            return super().peg_vertices(geo, params)
        # 正多边形内缩间隙 c 等价于外接圆半径减 c/cos(π/n)
        c = geo.g.clearance / 2 / math.cos(math.pi/self.rotation_order)
        return self._polygon(geo.g.radius*params[:, 0] - c)

    def opening_contains(self, geo: PreparedGeometry, params: torch.Tensor,
                         points_xy: torch.Tensor) -> torch.Tensor:
        if not geo.g.randomized:
            return super().opening_contains(geo, params, points_xy)
        return points_in_polygons(points_xy, self._polygon(self.opening_radius(geo, params)))

    def opening_radius(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        return self._apothem(geo.g, params)/math.cos(math.pi/self.rotation_order)

    def rim_yaws(self, g: Geometry) -> tuple[float, ...]:
        sides = self.rotation_order
        return tuple(math.pi + (i + 0.5)*2*math.pi/sides for i in range(sides))

    def rim_boxes(self, geo: PreparedGeometry, params: torch.Tensor,
                  width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 每条边向外一块盒体，沿边两端各延长 w·tan(π/n) 使相邻盒体在外角恰好合拢
        sides, half_angle = self.rotation_order, math.pi/self.rotation_order
        inner = self._apothem(geo.g, params)
        along = (inner + width)*math.tan(half_angle)
        half = torch.stack((along, width/2), -1)[:, None].expand(-1, sides, -1)
        return (inner + width/2)[:, None, None]*self._normals(params), half


class BoxShape(InsertionShape):
    """孔板与插头全部由轴对齐盒体拼成，随机化时逐环境改写盒体尺寸与位置。

    子类以 params 给出各盒体的 XY 边界，盒体的 Z 范围在建模时确定。
    """

    @abstractmethod
    def plate_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """孔板盒体名 -> (下界 (n, 2), 上界 (n, 2))"""

    @abstractmethod
    def peg_bounds(self, g: Geometry, params: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """插头盒体名 -> (下界 (n, 2), 上界 (n, 2))，坐标相对插头尖端"""

    @abstractmethod
    def hole_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        """孔口多边形 (n, V, 2)"""

    @abstractmethod
    def peg_outline(self, g: Geometry, params: torch.Tensor) -> torch.Tensor:
        """插头截面顶点 (n, V, 2)"""

    @abstractmethod
    def rim_bounds(self, g: Geometry, params: torch.Tensor,
                   width: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """围框盒体 (下界 (n, 2), 上界 (n, 2)) 列表"""

    def _nominal(self, g: Geometry) -> torch.Tensor:
        return torch.tensor([(*self.nominal_params(g), g.clearance)], dtype=torch.float64)

    def add_plate_colliders(self, hole: ET.Element, assets: ET.Element, g: Geometry) -> None:
        for name, (low, high) in self.plate_bounds(g, self._nominal(g)).items():
            add_box(hole, name, low[0].numpy(), high[0].numpy(), -g.depth, 0, "0.5 0.5 0.6 1")

    def add_peg_colliders(self, body: ET.Element, assets: ET.Element, g: Geometry,
                          tip_z: float) -> None:
        for name, (low, high) in self.peg_bounds(g, self._nominal(g)).items():
            add_box(body, name, low[0].numpy(), high[0].numpy(), tip_z, tip_z + g.peg_length,
                    "0.8 0.6 0.1 1")

    def apply_params(self, plant, ids: torch.Tensor, params: torch.Tensor) -> None:
        g = plant.geometry
        for bounds in (self.plate_bounds(g, params), self.peg_bounds(g, params)):
            for name, (low, high) in bounds.items():
                plant.set_box(ids, name, *box_from_bounds(low, high))

    def peg_vertices(self, geo: PreparedGeometry, params: torch.Tensor) -> torch.Tensor:
        return self.peg_outline(geo.g, params) if geo.g.randomized else super().peg_vertices(geo, params)

    def opening_contains(self, geo: PreparedGeometry, params: torch.Tensor,
                         points_xy: torch.Tensor) -> torch.Tensor:
        if not geo.g.randomized:
            return super().opening_contains(geo, params, points_xy)
        return points_in_polygons(points_xy, self.hole_outline(geo.g, params))

    def rim_yaws(self, g: Geometry) -> tuple[float, ...]:
        return (0.0,) * len(self.rim_bounds(g, self._nominal(g), torch.ones(1, dtype=torch.float64)))

    def rim_boxes(self, geo: PreparedGeometry, params: torch.Tensor,
                  width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = [box_from_bounds(low, high) for low, high in self.rim_bounds(geo.g, params, width)]
        return torch.stack([c for c, _ in boxes], 1), torch.stack([h for _, h in boxes], 1)
