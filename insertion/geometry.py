"""参数化凸碰撞几何生成，间隙定义为相对两侧面法向间距之差"""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

from mujoco._structs import MjModel
import numpy as np

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
             circle_sides: int | None = None) -> Geometry:
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
    return Geometry(shape, r, clearance, HOLE_DEPTH_M * scale, 0.035 * scale, p, peg, error)


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
    r, outer = g.radius, g.radius * 2
    if g.shape == "square":
        a = r / math.sqrt(2)
        # Tile the square annulus without overlapping wall volumes.
        bounds = (
            ([-2*a, -2*a], [-a, 2*a]),
            ([a, -2*a], [2*a, 2*a]),
            ([-a, -2*a], [a, -a]),
            ([-a, a], [a, 2*a]),
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
    return ET.tostring(root, encoding="unicode")


def build_model(g: Geometry) -> MjModel:
    return MjModel.from_xml_string(model_xml(g))


def symmetry_count(shape: str) -> int:
    return {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6,
            "l": 1, "circle": 0}[shape]
