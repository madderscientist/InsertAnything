"""参数化凸碰撞几何生成，间隙定义为相对两侧面法向间距之差"""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

from mujoco._structs import MjModel
import numpy as np

from .config import HOLE_DEPTH_M
from .shapes import ALL_SHAPES, InsertionShape, get_shape
from .shapes.base import RoughTile as RoughTile, inset_polygon as inset_polygon
from .shapes.base import points_in_polygon as points_in_polygon


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


    @property
    def definition(self) -> InsertionShape:
        return get_shape(self.shape)


def geometry(shape: str, clearance: float = 0.0001, scale: float = 1,
             circle_sides: int | None = None, plate_outer_radius_m: float = 0.0,
             rough_surface: bool = False, hole_depth_m: float = HOLE_DEPTH_M) -> Geometry:
    if shape not in ALL_SHAPES:
        raise ValueError(f"unsupported shape {shape}")
    r, c = 0.01 * scale, clearance / 2
    if scale <= 0 or not 0 < clearance < r / 2:
        raise ValueError("scale must be positive and clearance small relative to the peg")
    p, peg, error = get_shape(shape).cross_sections(r, c, circle_sides)
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
    definition = g.definition
    p = definition.rough_ring_polygon(g)
    tiles = list(definition.rough_infill_tiles(g, overlap=overlap, half_height=sz, margin=margin))

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
    definition = g.definition
    definition.add_plate_colliders(hole, assets, g)
    body = ET.SubElement(world, "body", name="tool", pos="0 0 0.06")
    ET.SubElement(body, "freejoint", name="tool_free")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1",
                  diaginertia="0.003 0.003 0.002")
    # 刚体原点为等效质心，位于名义插头尖端上方 50 mm
    tip_z = -0.05 * g.radius / 0.01
    definition.add_peg_colliders(body, assets, g, tip_z)
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
    return get_shape(shape).rotation_order
