"""参数化凸碰撞几何生成，间隙定义为相对两侧面法向间距之差"""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

from mujoco._structs import MjModel
import numpy as np
import torch

from .config import ALL_SHAPES, HOLE_DEPTH_M


@dataclass(frozen=True)
class Geometry:
    shape: str
    radius: float
    clearance: float
    depth: float
    peg_length: float
    hole_polygon: np.ndarray
    peg_polygon: np.ndarray
    circle_error: float = 0.0
    plate_outer_radius_m: float = 0.0
    rough_surface: bool = False


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


def geometry(shape: str, clearance: float = 0.0001, scale: float = 1,
             circle_sides: int | None = None, plate_outer_radius_m: float = 0.0,
             rough_surface: bool = False, hole_depth_m: float = HOLE_DEPTH_M) -> Geometry:
    if shape not in ALL_SHAPES:
        raise ValueError(f"unsupported shape {shape}")
    r, c = 0.01 * scale, clearance / 2
    if scale <= 0 or not 0 < clearance < r / 2:
        raise ValueError("scale must be positive and clearance small relative to the peg")
    error = 0.0
    if shape == "l":
        p = r * np.array([[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]])
        peg = np.array([[-r+c, -r+c], [r-c, -r+c], [r-c, -c],
                        [-c, -c], [-c, r-c], [-r+c, r-c]])
    else:
        sides = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}
        n = sides.get(shape, 0)
        if shape == "circle":
            # 控制圆形多边形孔壁的弦高误差不超过单侧间隙的 5%
            if circle_sides is not None and (
                    type(circle_sides) is not int or circle_sides < 3):
                raise ValueError("circle_sides must be an integer >= 3 or None")
            n = (circle_sides if circle_sides is not None else
                 max(128, math.ceil(math.pi / math.acos(1 - c * 0.05 / r))))
            error = r * (1 - math.cos(math.pi / n))
            if error >= c:
                raise ValueError("circle polygon approximation closes the radial clearance")
        angles = np.arange(n) * 2 * math.pi / n + math.pi / 2
        p = r * np.column_stack((np.cos(angles), np.sin(angles)))
        peg = inset_polygon(p, c)
    if not math.isfinite(plate_outer_radius_m) or plate_outer_radius_m < 0:
        raise ValueError("plate outer radius must be finite and nonnegative")
    if not isinstance(rough_surface, bool):
        raise ValueError("rough_surface must be a boolean")
    if not math.isfinite(hole_depth_m) or hole_depth_m <= 0:
        raise ValueError("hole_depth_m must be finite and positive")
    return Geometry(shape, r, clearance, hole_depth_m * scale, 0.035 * scale,
                    p, peg, error, plate_outer_radius_m, rough_surface)


def rough_surface_tiles(g: Geometry) -> tuple[RoughTile, ...]:
    """在孔口外侧生成细碎无裂缝且严格不侵入孔道的扇环微砖块地形贴片"""
    if not g.rough_surface:
        return ()
    scale = g.radius / 0.01
    r_max = 0.044 * scale
    ring_steps = tuple(x * scale for x in (
        0.0, 0.0040, 0.0082, 0.0128, 0.0178, 0.0235, 0.0300, 0.0370, 0.0440
    ))
    target_tile = 0.0048 * scale
    overlap = 0.0003 * scale
    sz = 0.0025 * scale
    margin = 0.00012 * scale
    tiles: list[RoughTile] = []
    if g.shape == "l":
        p = g.radius * np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
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
                    0.5 * (xb1 - xb0), 0.5 * (yb1 - yb0), sz, 0.0,
                    ((1.0, 0.0, margin), (0.0, 1.0, margin)),
                ))
    elif g.shape == "circle":
        n = 12
        angles = np.arange(n) * 2 * math.pi / n
        p = (g.radius / math.cos(math.pi / n)) * np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        p = g.hole_polygon

    for e in range(len(p)):
        p0, p1 = p[e], p[(e + 1) % len(p)]
        t_hat = (p1 - p0) / np.linalg.norm(p1 - p0)
        n_hat = np.array([t_hat[1], -t_hat[0]])
        if np.dot(n_hat, 0.5 * (p0 + p1)) < 0:
            n_hat, t_hat = -n_hat, -t_hat
        d_e = float(np.dot(n_hat, p0))
        v0_base, v1_base = sorted((float(np.dot(t_hat, p0)), float(np.dot(t_hat, p1))))
        yaw = math.atan2(float(n_hat[1]), float(n_hat[0]))
        guard = ((float(n_hat[0]), float(n_hat[1]), d_e + margin),)
        for ri in range(len(ring_steps) - 1):
            u0 = d_e + ring_steps[ri]
            u1 = min(d_e + ring_steps[ri + 1], r_max + 0.003 * scale)
            if u0 >= r_max:
                continue
            ub0 = d_e + margin if ri == 0 else u0 - overlap
            ub1 = u1 + overlap
            u_mid, su = 0.5 * (ub0 + ub1), 0.5 * (ub1 - ub0)
            v_cap = math.sqrt(max(0.0, (r_max + 0.003 * scale) ** 2 - u0 ** 2))
            va = max(v0_base * (u1 / d_e), -v_cap)
            vb = min(v1_base * (u1 / d_e), v_cap)
            if vb <= va + 1e-4:
                continue
            n_sub = max(1, round((vb - va) / target_tile))
            dv = (vb - va) / n_sub
            for si in range(n_sub):
                vc = va + (si + 0.5) * dv
                xy = u_mid * n_hat + vc * t_hat
                tiles.append(RoughTile(
                    float(xy[0]), float(xy[1]), su, 0.5 * dv + overlap, sz, yaw, guard,
                ))
    return tuple(tiles)


def _box(parent: ET.Element, name: str, low: np.ndarray, high: np.ndarray,
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


def _prism(parent: ET.Element, assets: ET.Element, name: str, polygon: np.ndarray,
           z_low: float, z_high: float, color: str) -> None:
    vertices = np.concatenate((np.column_stack((polygon, np.full(len(polygon), z_low))),
                               np.column_stack((polygon, np.full(len(polygon), z_high)))))
    ET.SubElement(assets, "mesh", name=name, vertex=" ".join(map(str, vertices.ravel())))
    ET.SubElement(parent, "geom", name=name, type="mesh", mesh=name, rgba=color)


def model_xml(g: Geometry) -> str:
    root = ET.Element("mujoco", model=f"insertion_{g.shape}")
    ET.SubElement(root, "compiler", angle="radian")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction="0.4 0.005 0.0001", condim="3",
                  margin="0", gap="0", solref="0.002 1", solimp="0.99 0.999 0.00001")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 0.2")
    hole = ET.SubElement(world, "body", name="hole")
    r, outer = g.radius, g.plate_outer_radius_m or g.radius * 2
    if g.shape == "square":
        a = r / math.sqrt(2)
        b = g.plate_outer_radius_m or 2*a
        # Tile the square annulus without overlapping wall volumes.
        bounds = (
            ([-b, -b], [-a, b]),
            ([a, -b], [b, b]),
            ([-a, -b], [a, -a]),
            ([-a, a], [a, b]),
        )
        for i, (low, high) in enumerate(bounds):
            _box(hole, f"wall_{i}", np.array(low), np.array(high),
                 -g.depth, 0, "0.5 0.5 0.6 1", yaw=math.pi/4)
    elif g.shape == "l":
        _box(hole, "wall_left", np.array([-outer, -outer]), np.array([-r, outer]),
             -g.depth, 0, "0.5 0.5 0.6 1")
        _box(hole, "wall_right", np.array([r, -outer]), np.array([outer, outer]),
             -g.depth, 0, "0.5 0.5 0.6 1")
        _box(hole, "wall_bottom", np.array([-r, -outer]), np.array([r, -r]),
             -g.depth, 0, "0.5 0.5 0.6 1")
        _box(hole, "wall_top", np.array([-r, r]), np.array([r, outer]),
             -g.depth, 0, "0.5 0.5 0.6 1")
        _box(hole, "wall_notch", np.array([0, 0]), np.array([r, r]),
             -g.depth, 0, "0.5 0.5 0.6 1")
    else:
        # 孔壁拆分为独立凸棱柱，避免整体凸包将中心孔封死
        p, out = g.hole_polygon, g.hole_polygon * 2
        if g.plate_outer_radius_m:
            edges = np.roll(p, -1, axis=0)-p
            inradius = np.min(np.abs(edges[:, 0]*p[:, 1]-edges[:, 1]*p[:, 0])
                              / np.linalg.norm(edges, axis=1))
            out = p*(g.plate_outer_radius_m/inradius)
        for i in range(len(p)):
            j = (i + 1) % len(p)
            _prism(hole, assets, f"wall_{i}", np.array([p[i], out[i], out[j], p[j]]),
                   -g.depth, 0, "0.5 0.5 0.6 1")
    body = ET.SubElement(world, "body", name="tool", pos="0 0 0.06")
    ET.SubElement(body, "freejoint", name="tool_free")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1",
                  diaginertia="0.003 0.003 0.002")
    # 刚体原点为等效质心，位于名义插头尖端上方 50 mm
    tip_z = -0.05 * g.radius / 0.01
    if g.shape == "square":
        a = r / math.sqrt(2) - g.clearance / 2
        _box(body, "peg", np.array([-a, -a]), np.array([a, a]),
             tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1", yaw=math.pi/4)
    elif g.shape == "l":
        c = g.clearance / 2
        _box(body, "peg_bottom", np.array([-r+c, -r+c]), np.array([r-c, -c]),
             tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1")
        _box(body, "peg_left", np.array([-r+c, -c]), np.array([-c, r-c]),
             tip_z, tip_z + g.peg_length, "0.8 0.6 0.1 1")
    elif g.shape == "circle":
        ET.SubElement(body, "geom", name="peg", type="cylinder",
                      pos=f"0 0 {tip_z + g.peg_length / 2}",
                      size=f"{r-g.clearance/2} {g.peg_length/2}", rgba="0.8 0.6 0.1 1")
    else:
        _prism(body, assets, "peg", g.peg_polygon, tip_z, tip_z + g.peg_length,
               "0.8 0.6 0.1 1")
    ET.SubElement(body, "site", name="tip", pos=f"0 0 {tip_z}", size="0.0005")
    ET.SubElement(body, "site", name="sensor", pos="0 0 0.03", size="0.001")
    if g.rough_surface:
        for hole_geom in hole.findall("geom"):
            hole_geom.set("contype", "1")
            hole_geom.set("conaffinity", "2")
        for tool_geom in body.findall("geom"):
            tool_geom.set("contype", "2")
            tool_geom.set("conaffinity", "1")
        surface = ET.SubElement(world, "body", name="surface", mocap="true", pos="0 0 0")
        for idx, tile in enumerate(rough_surface_tiles(g)):
            ET.SubElement(
                surface, "geom", name=f"surf_{idx}", type="box",
                contype="1", conaffinity="2",
                pos=f"{tile.cx} {tile.cy} {-tile.sz}",
                size=f"{tile.sx} {tile.sy} {tile.sz}",
                quat=f"{math.cos(tile.yaw / 2)} 0 0 {math.sin(tile.yaw / 2)}",
                rgba="0.46 0.49 0.58 1",
            )
    return ET.tostring(root, encoding="unicode")


def build_model(g: Geometry) -> MjModel:
    return MjModel.from_xml_string(model_xml(g))


def symmetry_count(shape: str) -> int:
    return {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6,
            "l": 1, "circle": 0}[shape]
