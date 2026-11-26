"""实时训练快照与最新策略回放可视化，不包含任何真机控制通信"""

import argparse
import base64
from dataclasses import replace
import io
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from mujoco._enums import mjtGeom, mjtObj
from mujoco._functions import mj_forward, mj_id2name, mj_resetData, mj_setConst, mju_mat2Quat, mju_rotVecQuat, mj_step
from mujoco._structs import MjData, MjModel
import numpy as np
import torch
from tensordict import TensorDict
import trimesh
import viser

from insertion import ALL_SHAPES, InsertionCfg, build_model, geometry
from insertion.env import PlantGroup
from insertion.search_map import CELL_M, GRID
from insertion.transforms import quat_rotvec, rotate, rotvec_quat, tip_pose


def replay_index(elapsed: float, count: int) -> int:
    if count < 1:
        raise ValueError("replay requires at least one frame")
    return int(max(0., elapsed) / 0.02) % count


def replay_config(environment: dict, shape: str, zero_noise: bool = False,
                  ignore_overload: bool = False, real_profile: bool = False,
                  training_level: bool = False) -> InsertionCfg:
    cfg = InsertionCfg.from_checkpoint(environment)
    if real_profile:
        from real2sim.evaluate_real2sim import REAL_PROFILE
        cfg = replace(cfg, **REAL_PROFILE)
    # training_level 保留训练态采样，由调用方按检查点课程等级设置难度、间隙与噪声
    cfg = replace(cfg,
                  shapes=(shape,), shape_env_counts=(), adaptive_sampling=False,
                  envs_per_shape=1, training=training_level,
                  terminate_on_overload=cfg.terminate_on_overload if training_level else not ignore_overload,
                  hard_state_reset_probability=0. if training_level else cfg.hard_state_reset_probability)
    if zero_noise:
        cfg = replace(cfg, noise_curriculum=False, noise_curriculum_mode="steps",
                      position_noise=0., rotation_noise=0., force_noise=0., moment_noise=0.,
                      position_bias=0., tip_calibration_error=0., grasp_position_error=0.,
                      grasp_z_error=0., sensor_rotation_error=0., rotation_bias=0.,
                      force_bias=0., moment_bias=0.)
    return cfg


class PolicyReplay:
    """Generate complete episodes separately, then display them at real-time speed."""

    def __init__(self, run_dir: Path, device: str, shape: str, zero_noise: bool = False,
                 ignore_overload: bool = False, real_profile: bool = False,
                 training_level: bool = False):
        self.run_dir, self.device, self.shape = run_dir, device, shape
        self.zero_noise, self.ignore_overload = zero_noise, ignore_overload
        self.training_level = training_level
        self.real_profile = real_profile
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.episode = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name="policy-replay", daemon=True)
        self.thread.start()

    def select(self, shape: str) -> None:
        with self.lock:
            self.shape = shape

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Policy replay failed; inspect viewer log") from self.error
            return self.episode

    def _run(self) -> None:
        try:
            from evaluate import load_checkpoint, REASONS
            from insertion import InsertionEnv

            torch.set_num_threads(1)
            env = None
            current_cfg = None
            while not self.stop.is_set():
                with self.lock:
                    shape = self.shape
                started = time.monotonic()
                metadata, actor = load_checkpoint(self.run_dir / "checkpoint.pt", self.device)
                cfg = replay_config(metadata["environment"], shape, self.zero_noise,
                                    self.ignore_overload, self.real_profile, self.training_level)
                if env is None or current_cfg != cfg:
                    env = InsertionEnv(cfg, self.device)
                    current_cfg = cfg
                group = env.groups[0]
                if self.training_level and env.noise_curriculum is not None:
                    curriculum = env.noise_curriculum
                    curriculum.load_levels(metadata["noise_curriculum_state"])
                    group.difficulty = curriculum.difficulty
                    scale = curriculum.scale_for(group.shape)
                    group.noise_scale = 0. if scale is None else scale
                    group.set_clearance(curriculum.clearance_for(group.shape))
                group.reset(torch.arange(1, device=self.device))
                if env.search_map is not None:
                    env.reset_search_map(torch.ones(1, dtype=torch.bool, device=self.device))
                    actor.search_map = env.search_map
                obs = env.get_observations()
                actor.reset()
                frames = []
                reason = "timeout"
                with torch.inference_mode():
                    for _ in range(env.max_episode_length):
                        if self.stop.is_set() or shape != self.shape:
                            break
                        frame = {
                            "qpos": group.sim.data.qpos[0].cpu().tolist(),
                            "grasp_offset": (group.offset[0]-group.nominal_offset[0]).cpu().tolist(),
                            "goal": group.goal[0].cpu().tolist(),
                            "wrench": group.wrench[0].cpu().tolist(),
                            "stiffness": group.k[0].cpu().tolist(),
                            "damping": group.d[0].cpu().tolist(),
                            "episode_step": int(group.length[0]),
                            "clearance": group.clearance,
                            "difficulty": group.difficulty,
                        }
                        if group.surf_slice is not None:
                            frame["surface_geom_pos"] = group.sim.model.geom_pos[0, group.surf_slice].cpu().tolist()
                            frame["surface_geom_quat"] = group.sim.model.geom_quat[0, group.surf_slice].cpu().tolist()
                        frames.append(frame)
                        obs, _, dones, info = env.step(actor(obs))
                        actor.reset(dones)
                        if bool(dones[0]):
                            reason = REASONS[int(info["terminal"]["reason"][0])]
                            break
                if self.stop.is_set():
                    return
                if shape != self.shape:
                    continue
                episode = {
                    "shape": shape, "frames": frames,
                    "policy_steps": metadata["progress"]["policy_steps"],
                    "generation_seconds": time.monotonic()-started,
                    "generated_at": time.time(), "reason": reason,
                    "zero_noise": self.zero_noise,
                }
                with self.lock:
                    self.episode = episode
                logging.info("Policy replay ready: shape=%s policy_steps=%d zero_noise=%s "
                             "result=%s frames=%d generation=%.2fs",
                             shape, episode["policy_steps"], self.zero_noise, reason,
                             len(frames), episode["generation_seconds"])
        except Exception as error:
            logging.exception("Policy replay worker failed; training is unaffected.")
            with self.lock:
                self.error = error

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            logging.error("Policy replay did not finish within its shutdown grace period.")


def geom_mesh(model: MjModel, index: int) -> trimesh.Trimesh:
    kind = model.geom_type[index]
    if kind == mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2*model.geom_size[index])
    if kind == mjtGeom.mjGEOM_CYLINDER:
        radius, half_height = model.geom_size[index, :2]
        return trimesh.creation.cylinder(radius=radius, height=2*half_height, sections=64)
    if kind == mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        vertex, face = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        return trimesh.Trimesh(
            vertices=model.mesh_vert[vertex:vertex+model.mesh_vertnum[mesh]].copy(),
            faces=model.mesh_face[face:face+model.mesh_facenum[mesh]].copy(),
            process=False,
        )
    raise ValueError(f"unsupported insertion geom type: {kind}")


def vector_arrow(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-12:
        return np.zeros((3, 2, 3), dtype=np.float32)
    direction = vector/length
    basis = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(direction, basis)
    side /= np.linalg.norm(side)
    head = min(length*0.3, 0.008)
    base = vector-head*direction
    return np.asarray([[np.zeros(3), vector],
                       [vector, base+head*0.5*side],
                       [vector, base-head*0.5*side]], dtype=np.float32)


def visible_wrench_vector(value: np.ndarray, reference: float) -> np.ndarray:
    return value*(0.03/reference)


def terrain_height_color(height_m: float, ref_max_m: float = 0.0025) -> tuple[int, int, int]:
    t = float(np.clip(height_m / max(ref_max_m, 1e-4), 0.0, 1.0))
    if t < 0.5:
        u = t * 2.0
        return (
            int(75 + u * (95 - 75)),
            int(102 + u * (165 - 102)),
            int(142 + u * (145 - 142)),
        )
    u = (t - 0.5) * 2.0
    return (
        int(95 + u * (225 - 95)),
        int(165 + u * (135 - 165)),
        int(145 + u * (70 - 145)),
    )


class SnapshotScene:
    def __init__(self, server: viser.ViserServer | Any, shape: str, cfg: InsertionCfg):
        self.server = server
        self.model = build_model(geometry(shape, cfg.clearance, cfg.scale, cfg.circle_sides,
                                          cfg.plate_outer_radius_m, cfg.rough_surface,
                                          cfg.hole_depth_m))
        self.data = MjData(self.model)
        self.base_geom_pos = self.model.geom_pos.copy()
        self.tool_id = self.model.body("tool").id
        self.peg_ids = np.flatnonzero(self.model.geom_bodyid == self.tool_id)
        self.surf_ids = (
            np.flatnonzero(self.model.geom_bodyid == self.model.body("surface").id)
            if cfg.rough_surface else np.array([], dtype=int))
        surf_set = set(self.surf_ids.tolist())
        peg_set = set(self.peg_ids.tolist())
        self.wall_ids = np.array(
            [i for i in range(self.model.ngeom) if i not in surf_set and i not in peg_set],
            dtype=int,
        )
        self.surf_local_meshes = [geom_mesh(self.model, int(i)) for i in self.surf_ids]
        self.last_surf_signature: bytes | None = None
        self.walls_placed = False
        self.terrain_handle: Any = None
        self.nominal_tip = np.array([0., 0., -0.05*cfg.scale])
        # 使用独立随机命名空间，防止切换形状时客户端残留上一几何的四元数姿态
        self.root = server.scene.add_frame(f"/plant/{shape}-{uuid4().hex}", show_axes=False)
        self.geom_handles: dict[int, Any] = {}
        for index in (*self.wall_ids.tolist(), *self.peg_ids.tolist()):
            mesh = geom_mesh(self.model, index)
            rgba = self.model.geom_rgba[index]
            name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, index)
            self.geom_handles[index] = server.scene.add_mesh_simple(
                f"{self.root.name}/{name}", mesh.vertices, mesh.faces,
                color=(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)),
                opacity=1.0 if index in peg_set else 0.38,
                side="double",
            )
        self.tip = server.scene.add_frame(f"{self.root.name}/actual_tip", axes_length=0.006, axes_radius=0.0002)
        self.nominal_tcp = server.scene.add_frame(f"{self.root.name}/nominal_tcp", axes_length=0.0045, axes_radius=0.00016)
        self.goal = server.scene.add_frame(f"{self.root.name}/true_goal", axes_length=0.006, axes_radius=0.0002)
        self.hole = server.scene.add_frame(f"{self.root.name}/hole", axes_length=0.015, axes_radius=0.0002)
        self.wrench_frame = server.scene.add_frame(
            f"{self.root.name}/wrench", show_axes=False)
        self.wrench_vectors = []
        for name, color in (("force", (255, 80, 60)), ("torque", (80, 140, 255))):
            self.wrench_vectors.append(server.scene.add_line_segments(
                f"{self.wrench_frame.name}/{name}", vector_arrow(np.zeros(3)),
                color, thickness=6., thickness_units="screen",
                position=(0., 0.008 if name == "force" else -0.008, 0.)))

    def _build_combined_terrain_mesh(self, surf_pos: np.ndarray) -> trimesh.Trimesh:
        heights = surf_pos[:, 2] + self.model.geom_size[self.surf_ids, 2]
        span = max(float(heights.max()), 0.0025)
        verts_all, faces_all, colors_all = [], [], []
        v_offset = 0
        for local_i, geom_i in enumerate(self.surf_ids):
            base = self.surf_local_meshes[local_i]
            rot = self.data.geom_xmat[geom_i].reshape(3, 3)
            pos = self.data.geom_xpos[geom_i]
            verts = base.vertices @ rot.T + pos
            rgb = terrain_height_color(float(heights[local_i]), span)
            rgba = np.empty((len(verts), 4), dtype=np.uint8)
            rgba[:, 0], rgba[:, 1], rgba[:, 2], rgba[:, 3] = rgb[0], rgb[1], rgb[2], 235
            verts_all.append(verts)
            faces_all.append(base.faces + v_offset)
            colors_all.append(rgba)
            v_offset += len(verts)
        return trimesh.Trimesh(
            vertices=np.vstack(verts_all),
            faces=np.vstack(faces_all),
            vertex_colors=np.vstack(colors_all),
            process=False,
        )

    def update(self, plant: dict) -> None:
        wrench = np.asarray(plant["wrench"], dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("viewer wrench must contain six finite components")
        self.data.qpos[:] = plant["qpos"]
        self.model.geom_pos[self.peg_ids] = (
            self.base_geom_pos[self.peg_ids] + np.asarray(plant["grasp_offset"]))
        rebuild_terrain = False
        surf_pos = None
        if len(self.surf_ids) and "surface_geom_pos" in plant:
            surf_pos = np.asarray(plant["surface_geom_pos"], dtype=np.float32)
            surf_quat = np.asarray(plant["surface_geom_quat"], dtype=np.float32)
            sig = surf_pos.tobytes() + surf_quat.tobytes()
            if sig != self.last_surf_signature:
                self.last_surf_signature = sig
                self.model.geom_pos[self.surf_ids] = surf_pos
                self.model.geom_quat[self.surf_ids] = surf_quat
                rebuild_terrain = True
        mj_forward(self.model, self.data)
        combined_terrain = (
            self._build_combined_terrain_mesh(surf_pos)
            if rebuild_terrain and surf_pos is not None else None
        )
        active_indices = (
            (*self.wall_ids.tolist(), *self.peg_ids.tolist())
            if not self.walls_placed else tuple(self.peg_ids.tolist())
        )
        with self.server.atomic():
            if combined_terrain is not None:
                self.terrain_handle = self.server.scene.add_mesh_trimesh(
                    f"{self.root.name}/rough_terrain", combined_terrain)
            for index in active_indices:
                handle = self.geom_handles[index]
                q = np.empty(4)
                mju_mat2Quat(q, self.data.geom_xmat[index])
                handle.position = self.data.geom_xpos[index]
                handle.wxyz = q
            self.walls_placed = True
            rotated = np.empty(3)
            mju_rotVecQuat(rotated, self.nominal_tip+plant["grasp_offset"],
                           self.data.qpos[3:7])
            self.tip.position = self.data.qpos[:3]+rotated
            self.tip.wxyz = self.data.qpos[3:7]
            calib_off = np.asarray(plant.get("calib_offset", (0., 0., 0.)), dtype=float)
            mju_rotVecQuat(rotated, self.nominal_tip + calib_off, self.data.qpos[3:7])
            self.nominal_tcp.position = self.data.qpos[:3] + rotated
            self.nominal_tcp.wxyz = self.data.qpos[3:7]
            self.goal.position = np.asarray(plant["goal"])
            mju_rotVecQuat(rotated, self.nominal_tip, self.data.qpos[3:7])
            # 偏移显示避免不透明插头遮挡；只表达载荷方向，不代表实际作用点
            self.wrench_frame.position = self.data.qpos[:3]+rotated+np.array([0.03, 0., 0.02])
            self.wrench_frame.wxyz = self.data.qpos[3:7]
            for handle, value, reference in zip(
                    self.wrench_vectors, (wrench[:3], wrench[3:]), (5., 0.1), strict=True):
                handle.points = vector_arrow(visible_wrench_vector(value, reference))

    def remove(self) -> None:
        self.root.remove()


class CpuSimData:
    def __init__(self, model: MjModel, data: MjData):
        self._d = data
        self.qpos = torch.zeros(1, model.nq, dtype=torch.float32)
        self.qvel = torch.zeros(1, model.nv, dtype=torch.float32)
        self.qacc = torch.zeros(1, model.nv, dtype=torch.float32)
        self.qfrc_constraint = torch.zeros(1, model.nv, dtype=torch.float32)
        self.xfrc_applied = torch.zeros(1, model.nbody, 6, dtype=torch.float32)

    def pull(self) -> None:
        self.qpos[0] = torch.from_numpy(self._d.qpos)
        self.qvel[0] = torch.from_numpy(self._d.qvel)
        self.qacc[0] = torch.from_numpy(self._d.qacc)
        self.qfrc_constraint[0] = torch.from_numpy(self._d.qfrc_constraint)

    def push_state(self) -> None:
        self._d.qpos[:] = self.qpos[0].numpy()
        self._d.qvel[:] = self.qvel[0].numpy()
        self._d.xfrc_applied[:] = self.xfrc_applied[0].numpy()

    def push_wrench(self) -> None:
        self._d.xfrc_applied[:] = self.xfrc_applied[0].numpy()


class CpuSimModel:
    def __init__(self, model: MjModel):
        self._m = model
        self.body_mass = torch.from_numpy(model.body_mass).float().unsqueeze(0).clone()
        self.body_inertia = torch.from_numpy(model.body_inertia).float().unsqueeze(0).clone()
        self.geom_friction = torch.from_numpy(model.geom_friction).float().unsqueeze(0).clone()
        self.geom_pos = torch.from_numpy(model.geom_pos).float().unsqueeze(0).clone()
        self.geom_quat = torch.from_numpy(model.geom_quat).float().unsqueeze(0).clone()

    def push(self) -> None:
        self._m.body_mass[:] = self.body_mass[0].numpy()
        self._m.body_inertia[:] = self.body_inertia[0].numpy()
        self._m.geom_friction[:] = self.geom_friction[0].numpy()
        self._m.geom_pos[:] = self.geom_pos[0].numpy()
        self._m.geom_quat[:] = self.geom_quat[0].numpy()


class CpuSim:
    def __init__(self, model: MjModel):
        self.mj_model = model
        self.mj_data = MjData(model)
        self.model = CpuSimModel(model)
        self.data = CpuSimData(model, self.mj_data)
        self.use_cuda_graph = False
        self.step_graph = None
        self.nan_guard = type("NanGuard", (), {"enabled": False})()

    def reset(self, ids: torch.Tensor) -> None:
        del ids
        mj_resetData(self.mj_model, self.mj_data)
        self.data.xfrc_applied.zero_()
        self.data.pull()

    def recompute_constants(self, level: Any) -> None:
        del level
        self.model.push()
        mj_setConst(self.mj_model, self.mj_data)

    def forward(self) -> None:
        self.model.push()
        self.data.push_state()
        mj_forward(self.mj_model, self.mj_data)
        self.data.pull()

    def step(self) -> None:
        self.data.push_wrench()
        mj_step(self.mj_model, self.mj_data)
        self.data.pull()


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.float64)
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]
    return out


def _np_quat_canonical(q: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]))
    q = q / max(n, 1e-12)
    for i in range(4):
        if abs(q[i]) > 1e-8:
            return -q if q[i] < 0 else q
    return q


def _np_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    uvx = qy * vz - qz * vy
    uvy = qz * vx - qx * vz
    uvz = qx * vy - qy * vx
    out = np.empty(3, dtype=np.float64)
    out[0] = vx + 2.0 * (qw * uvx + (qy * uvz - qz * uvy))
    out[1] = vy + 2.0 * (qw * uvy + (qz * uvx - qx * uvz))
    out[2] = vz + 2.0 * (qw * uvz + (qx * uvy - qy * uvx))
    return out


def _np_quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    out = np.empty(4, dtype=np.float64)
    out[0] = aw * bw - (ax * bx + ay * by + az * bz)
    out[1] = aw * bx + bw * ax + (ay * bz - az * by)
    out[2] = aw * by + bw * ay + (az * bx - ax * bz)
    out[3] = aw * bz + bw * az + (ax * by - ay * bx)
    return out


def _np_quat_rotvec(q: np.ndarray) -> np.ndarray:
    q = _np_quat_canonical(q)
    sine = math.sqrt(float(q[1]*q[1] + q[2]*q[2] + q[3]*q[3]))
    if sine > 1e-7:
        angle = 2.0 * math.atan2(sine, float(q[0]))
        return q[1:] * (angle / sine)
    return q[1:] * 2.0


def _np_transform_wrench(f: np.ndarray, m: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    fr = _np_rotate(q, f)
    mr = _np_rotate(q, m)
    out = np.empty(6, dtype=np.float64)
    out[:3] = fr
    out[3:] = mr - _cross3(r, fr)
    return out


class RealtimePlant(PlantGroup):
    """Single-environment real-time CPU MuJoCo plant with interactive pose/noise/dynamics control."""

    def __init__(self, shape: str, cfg: InsertionCfg):
        cfg = replace(cfg, shapes=(shape,), envs_per_shape=1, shape_env_counts=(),
                      adaptive_sampling=False, training=False, episode_seconds=1e9,
                      enable_timeout=False, rough_surface=True,
                      physics_cuda_graph=False, observation_cuda_graph=False)
        super().__init__(shape, cfg, "cpu")
        seed_now = time.time_ns() & 0x7FFF_FFFF
        self.generator.manual_seed(seed_now)
        self.walk_generator.manual_seed((seed_now + 500003) & 0x7FFF_FFFF)
        self.grasp_dir = torch.zeros(1, 3)
        self.calib_dir = torch.zeros(1, 3)
        self.pos_bias_dir = torch.zeros(1, 3)
        self.rot_bias_dir = torch.zeros(1, 3)
        self.sensor_rot_dir = torch.zeros(1, 3)
        self.force_bias_dir = torch.zeros(1, 3)
        self.moment_bias_dir = torch.zeros(1, 3)
        self.sampled_act_delay = 0
        self.wave_u_wl = torch.zeros(1, cfg.rough_num_waves)
        self.wave_theta = torch.zeros(1, cfg.rough_num_waves)
        self.wave_phase = torch.zeros(1, cfg.rough_num_waves)
        self.wave_weights = torch.ones(1, cfg.rough_num_waves) / cfg.rough_num_waves
        self.tile_jitter = torch.zeros(1, max(1, len(self.surf_geom_ids)))
        self.terrain_stepped = False
        self.resample_fixed_directions()
        self.resample_terrain_waves()

    def _build_simulation(self) -> None:
        cfg = self.cfg
        self.geometry = geometry(self.shape, self.clearance, cfg.scale, cfg.circle_sides,
                                 cfg.plate_outer_radius_m, cfg.rough_surface, cfg.hole_depth_m)
        self.model = build_model(self.geometry)
        self.model.opt.timestep = cfg.physics_dt
        self.model.opt.gravity[:] = 0
        self.model.opt.iterations = 50
        self.model.opt.ls_iterations = 20
        self.model.bvh_aabb[:, 3:] += 0.06
        self.body_id = self.model.body("tool").id
        self.sim = CpuSim(self.model)
        self.peg_geom_ids = (self.model.geom_bodyid == self.body_id).nonzero()[0].tolist()
        self.default_geom_pos = torch.tensor(
            self.model.geom_pos, device=self.device, dtype=torch.float32)
        self.peg_polygon = torch.tensor(
            self.geometry.peg_polygon, dtype=torch.float32, device=self.device)
        self.hole_polygon = torch.tensor(
            self.geometry.hole_polygon, dtype=torch.float32, device=self.device)
        self._init_rough_surface()

    def resample_fixed_directions(self) -> None:
        self.grasp_dir = torch.randn(1, 3, generator=self.generator)
        self.calib_dir = self.uniform(1, 3, -1., 1.)
        self.pos_bias_dir = self.uniform(1, 3, -1., 1.)
        self.rot_bias_dir = self.uniform(1, 3, -1., 1.)
        self.sensor_rot_dir = self.uniform(1, 3, -1., 1.)
        self.force_bias_dir = self.sample_wrench_bias(1, 1.0)
        self.moment_bias_dir = self.sample_wrench_bias(1, 1.0)
        self.sampled_act_delay = int(torch.randint(0, 3, (1,), generator=self.generator).item())
        self.gain_q = rotvec_quat(self.uniform(1, 3, -0.5, 0.5))
        self.damping_ratio = self.uniform(1, 6, 0.8, 1.3)
        self.mass = self.uniform(1, 1, 0.7, 1.5)
        inertia_parts = self.uniform(1, 3, 0.001, 0.0025)
        self.inertia = (inertia_parts.sum(-1, keepdim=True) - inertia_parts) * self.cfg.scale ** 2
        self.down_mass = self.uniform(1, 1, 0.2, 0.4)
        down_parts = self.uniform(1, 3, 0.00015, 0.00035)
        self.down_inertia = (down_parts.sum(-1, keepdim=True) - down_parts) * self.cfg.scale ** 2
        self.sim.model.body_mass[0, self.body_id] = self.mass[0, 0]
        self.sim.model.body_inertia[0, self.body_id] = self.inertia[0]

    def resample_terrain_waves(self) -> None:
        m = self.cfg.rough_num_waves
        self.wave_u_wl = self.uniform(1, m, 0., 1.)
        self.wave_theta = self.uniform(1, m, -3.14159265, 3.14159265)
        self.wave_phase = self.uniform(1, m, -3.14159265, 3.14159265)
        w = self.uniform(1, m, 0.2, 1.0)
        self.wave_weights = w / w.sum(dim=-1, keepdim=True)
        k_tiles = max(1, len(self.surf_geom_ids))
        self.tile_jitter = self.uniform(1, k_tiles, -0.5, 0.5)
        self.terrain_stepped = bool(self.uniform(1, 1, 0., 1.).item() < self.cfg.rough_stepped_prob)

    def apply_terrain(self, amplitude_mm: float, wl_min_mm: float, wl_max_mm: float) -> None:
        if self.surf_slice is None:
            return
        lo = max(4.0, min(wl_min_mm, wl_max_mm)) * 1e-3 * self.cfg.scale
        hi = max(lo + 1e-4, max(wl_min_mm, wl_max_mm) * 1e-3 * self.cfg.scale)
        wavelengths = (math.log(lo) + self.wave_u_wl * (math.log(hi) - math.log(lo))).exp()
        amp = torch.tensor([[max(0.0, amplitude_mm) * 1e-3 * self.cfg.scale]], dtype=torch.float32)
        jitter = self.tile_jitter if self.tile_jitter.shape[1] == len(self.surf_geom_ids) else None
        stepped = torch.tensor([[self.terrain_stepped]], dtype=torch.bool)
        pos, q_full, tile_top_z = self._compute_rough_tiles(
            amp, wavelengths, self.wave_theta, self.wave_phase, self.wave_weights, jitter, stepped)
        self.sim.model.geom_pos[:, self.surf_slice] = pos
        self.sim.model.geom_quat[:, self.surf_slice] = q_full
        self.surface_max_height[:] = tile_top_z.amax(dim=-1).clamp_min(0.0)
        self.sim.model.push()

    def apply_parameters(self, *, noise_scale: float, pos_bias_mm: float,
                         grasp_xy_err_mm: float, grasp_z_err_mm: float,
                         calib_err_mm: float, rot_bias_deg: float, pos_noise_mm: float, rot_noise_deg: float,
                         force_bias_n: float, force_noise_n: float, moment_bias_nm: float,
                         moment_noise_nm: float, k_xyz: float, k_rot: float, friction_mu: float,
                         stiction_f: float, stiction_m: float, wall_f: float,
                         terminate_on_overload: bool, recompute_geom: bool = True) -> None:
        self.noise_scale = float(noise_scale)
        self.cfg = replace(
            self.cfg,
            position_noise=pos_noise_mm * 1e-3,
            rotation_noise=math.radians(rot_noise_deg),
            force_noise=force_noise_n,
            moment_noise=moment_noise_nm,
            terminate_on_overload=terminate_on_overload,
        )
        grasp = torch.stack((
            self.grasp_dir[:, 0] * (grasp_xy_err_mm * 1e-3),
            self.grasp_dir[:, 1] * (grasp_xy_err_mm * 1e-3),
            self.grasp_dir[:, 2].clamp(-2.5, 2.5) * (grasp_z_err_mm * 1e-3),
        ), dim=-1) * self.noise_scale
        self.offset[0] = self.nominal_offset[0] + grasp[0]
        if recompute_geom:
            for geom_id in self.peg_geom_ids:
                self.sim.model.geom_pos[0, geom_id] = self.default_geom_pos[geom_id] + grasp[0]
        self.sim.model.geom_friction[0, :, 0] = float(friction_mu)
        if recompute_geom:
            self.sim.recompute_constants(None)
        else:
            self.sim.model.push()
        self.calibration[0] = self.calib_dir[0] * (calib_err_mm * 1e-3 * self.noise_scale)
        self.bias[0] = self.pos_bias_dir[0] * (pos_bias_mm * 1e-3 * self.noise_scale)
        if self.cfg.align_goal_z_to_observation:
            self.bias[0, 2] = self.calibration[0, 2] - grasp[0, 2]
        self.rotation_bias_q[0] = rotvec_quat(
            self.rot_bias_dir * (math.radians(rot_bias_deg) * self.noise_scale))[0]
        self.sensor_error[0] = rotvec_quat(
            self.sensor_rot_dir * (self.cfg.sensor_rotation_error * self.noise_scale))[0]
        self.wrench_bias[0, :3] = self.force_bias_dir[0] * (force_bias_n * self.noise_scale)
        self.wrench_bias[0, 3:] = self.moment_bias_dir[0] * (moment_bias_nm * self.noise_scale)
        self.k[0] = torch.tensor([k_xyz, k_xyz, k_xyz, k_rot, k_rot, k_rot], dtype=torch.float32)
        self.initial_k[0] = self.k[0].double()
        wf = self.cfg.impedance_walk_fraction if self.noise_scale > 0 else 0.0
        self.walk_lower[0] = self.initial_k[0] * (1.0 - wf)
        self.walk_upper[0] = self.initial_k[0] * (1.0 + wf)
        if recompute_geom:
            self.walk_current[0] = self.initial_k[0]
            self.walk_target[0] = self.initial_k[0]
        else:
            self.walk_current[0] = self.walk_current[0].clamp(self.walk_lower[0], self.walk_upper[0])
            self.walk_target[0] = self.walk_target[0].clamp(self.walk_lower[0], self.walk_upper[0])
        inertia6 = torch.cat((self.mass.expand(1, 3), self.inertia), -1)
        self.d[0] = (2.0 * self.damping_ratio * torch.sqrt(self.k * inertia6))[0]
        self.stiction[0] = torch.tensor([stiction_f]*3 + [stiction_m]*3, dtype=torch.float32)
        self.wall_friction[0] = float(wall_f)
        self.delay[0] = self.sampled_act_delay if self.noise_scale > 0 else 0

    def set_initial_pose(self, x_mm: float, y_mm: float, z_mm: float,
                         rx_deg: float, ry_deg: float, rz_deg: float,
                         depth_mm: float) -> None:
        self.sim.reset(torch.zeros(1, dtype=torch.long))
        self.goal[0] = torch.tensor([0., 0., -depth_mm * 1e-3], dtype=torch.float32)
        pos = torch.tensor([[x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3]], dtype=torch.float32)
        rv = torch.tensor([[math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)]],
                          dtype=torch.float32)
        q = rotvec_quat(rv)
        self.sim.data.qpos[0, :3] = (pos - rotate(q, self.offset))[0]
        self.sim.data.qpos[0, 3:7] = q[0]
        self.tip_ema[:] = torch.cat((pos[:, :2], pos[:, 2:3].clamp_max(0.0)), dim=-1)
        self.sim.data.qvel.zero_()
        self.sim.data.xfrc_applied.zero_()
        self.sim.forward()
        nominal_tip = tip_pose(self.sim.data.qpos[:, :3], q,
                               self.nominal_offset + self.calibration)
        self.targets_p[:] = nominal_tip.unsqueeze(0)
        self.targets_q[:] = q.unsqueeze(0)
        self.physics_p_goal[:] = nominal_tip
        self.physics_q_goal[:] = q
        self.delay[0] = self.sampled_act_delay if self.noise_scale > 0 else 0
        for buffer in (self.length, self.hold, self.peak_force, self.peak_moment,
                       self.step_peak_force, self.step_peak_moment,
                       self.last_action, self.wrench, self.raw_wrench, self.filtered_sensor,
                       self.hysteresis_sensor, self.wall_force_z):
            buffer.zero_()
        if self.cfg.position_observation_delay:
            self.reset_position_observation(torch.zeros(1, dtype=torch.long), nominal_tip)

    def advance_physics(self) -> None:
        cfg = self.cfg
        d = self.sim.mj_data
        m = self.model
        body_id = self.body_id
        gain_q = self.gain_q[0].numpy().astype(np.float64)
        gain_q_inv = np.array([gain_q[0], -gain_q[1], -gain_q[2], -gain_q[3]], dtype=np.float64)
        do_walk = (cfg.dynamics != "nominal" and cfg.impedance_walk_fraction > 0
                   and not torch.allclose(self.walk_lower, self.walk_upper))
        walk_cur = self.walk_current[0].numpy().copy()
        walk_tgt = self.walk_target[0].numpy()
        walk_init = self.initial_k[0].numpy()
        walk_lo = self.walk_lower[0].numpy()
        walk_hi = self.walk_upper[0].numpy()
        walk_alpha = 1.0 - math.exp(-cfg.physics_dt / cfg.impedance_walk_time_constant_seconds)
        walk_max_delta = walk_init * (cfg.impedance_walk_rate_per_second * cfg.physics_dt)
        damp_ratio = self.damping_ratio[0].numpy().astype(np.float64)
        inertia6 = np.concatenate((
            np.repeat(self.mass[0].numpy().astype(np.float64), 3),
            self.inertia[0].numpy().astype(np.float64),
        ))
        k_f = self.k[0, :3].numpy().astype(np.float64).copy()
        k_m = self.k[0, 3:].numpy().astype(np.float64).copy()
        d_f = self.d[0, :3].numpy().astype(np.float64).copy()
        d_m = self.d[0, 3:].numpy().astype(np.float64).copy()
        stic_f = self.stiction[0, :3].numpy().astype(np.float64)
        stic_m = self.stiction[0, 3:].numpy().astype(np.float64)
        nom_calib = (self.nominal_offset[0] + self.calibration[0]).numpy().astype(np.float64)
        offset = self.offset[0].numpy().astype(np.float64)
        sensor_off = self.sensor_offset[0].numpy().astype(np.float64)
        sensor_err = self.sensor_error[0].numpy().astype(np.float64)
        r_st_rot = _np_rotate(sensor_err, nom_calib - sensor_off)
        r_raw = offset - sensor_off
        q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        p_goal = self.physics_p_goal[0].numpy().astype(np.float64)
        q_goal = self.physics_q_goal[0].numpy().astype(np.float64)
        wall_fric = float(self.wall_friction[0])
        mass = float(self.mass[0, 0])
        down_mass = float(self.down_mass[0, 0])
        down_inertia = self.down_inertia[0].numpy().astype(np.float64)
        alpha = 1.0 - math.exp(-cfg.physics_dt / cfg.filter_seconds)
        hyst_hw = self.hysteresis_half_width.numpy().astype(np.float64)
        filt = self.filtered_sensor[0].numpy().astype(np.float64).copy()
        hyst = self.hysteresis_sensor[0].numpy().astype(np.float64).copy()
        step_pf = float(self.step_peak_force[0])
        step_pm = float(self.step_peak_moment[0])
        pf = float(self.peak_force[0])
        pm = float(self.peak_moment[0])
        r_sq = self.geometry.radius ** 2
        ramp_depth = cfg.inserted_load_depth_m
        wall_fz = 0.0
        wrench_out = np.zeros(6, dtype=np.float64)
        raw_out = np.zeros(6, dtype=np.float64)

        for _ in range(cfg.decimation):
            if do_walk:
                delta = np.clip((walk_tgt - walk_cur) * walk_alpha, -walk_max_delta, walk_max_delta)
                walk_cur = np.clip(walk_cur + delta, walk_lo, walk_hi)
                k_f = walk_cur[:3]
                k_m = walk_cur[3:]
                d_all = 2.0 * damp_ratio * np.sqrt(walk_cur * inertia6)
                d_f = d_all[:3]
                d_m = d_all[3:]
            pos = d.qpos[:3]
            q = _np_quat_canonical(d.qpos[3:7])
            v = d.qvel[:3]
            omega = _np_rotate(q, d.qvel[3:6])
            r_ct = _np_rotate(q, nom_calib)
            p_tip = pos + r_ct
            v_tip = v + _cross3(omega, r_ct)

            loc_ef = _np_rotate(gain_q_inv, p_goal - p_tip)
            loc_vf = _np_rotate(gain_q_inv, v_tip)
            spr_f = k_f * loc_ef
            spr_f = np.sign(spr_f) * np.maximum(np.abs(spr_f) - stic_f, 0.0)
            force = np.clip(_np_rotate(gain_q, spr_f - d_f * loc_vf), -40.0, 40.0)

            q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
            ang_err = _np_quat_rotvec(_np_quat_mul(q_goal, q_inv))
            loc_em = _np_rotate(gain_q_inv, ang_err)
            loc_vm = _np_rotate(gain_q_inv, omega)
            spr_m = k_m * loc_em
            spr_m = np.sign(spr_m) * np.maximum(np.abs(spr_m) - stic_m, 0.0)
            moment_tip = np.clip(_np_rotate(gain_q, spr_m - d_m * loc_vm), -3.0, 3.0)

            true_tip = pos + _np_rotate(q, offset)
            in_open = 1.0 if (true_tip[0] ** 2 + true_tip[1] ** 2 <= r_sq) else 0.0
            depth_scale = min(max((-true_tip[2] - 0.0005) / ramp_depth, 0.0), 1.0) * in_open
            max_wf = wall_fric * depth_scale
            wall_fz = min(max(-force[2] - mass * v[2] / 0.005, -max_wf), max_wf)

            d.xfrc_applied[body_id, 0] = force[0]
            d.xfrc_applied[body_id, 1] = force[1]
            d.xfrc_applied[body_id, 2] = force[2] + wall_fz
            d.xfrc_applied[body_id, 3:] = moment_tip + _cross3(r_ct, force)

            mj_step(m, d)

            q2 = _np_quat_canonical(d.qpos[3:7])
            q2_inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]], dtype=np.float64)
            acc = d.qacc[:3]
            alpha_loc = d.qacc[3:6]
            omega_loc = d.qvel[3:6]
            cf = np.array(
                [d.qfrc_constraint[0], d.qfrc_constraint[1], d.qfrc_constraint[2] + wall_fz],
                dtype=np.float64,
            )
            cm = _np_rotate(q2, d.qfrc_constraint[3:6])
            iner_m = _np_rotate(
                q2, down_inertia * alpha_loc + _cross3(omega_loc, down_inertia * omega_loc))
            wrist = _np_transform_wrench(cf - down_mass * acc, cm - iner_m, q2_inv, sensor_off)
            filt += alpha * (wrist - filt)
            hyst = np.minimum(np.maximum(hyst, filt - hyst_hw), filt + hyst_hw)
            wrench_out = _np_transform_wrench(hyst[:3], hyst[3:], sensor_err, r_st_rot)
            raw_out = _np_transform_wrench(wrist[:3], wrist[3:], q_ident, r_raw)

            f_norm = float(np.linalg.norm(raw_out[:3]))
            m_norm = float(np.linalg.norm(raw_out[3:]))
            if cfg.inserted_load_cost:
                tz = float((d.qpos[:3] + _np_rotate(q2, offset))[2])
                zw = min(max(1.0 + tz / cfg.inserted_load_depth_m, 0.0), 1.0)
                rhf = _np_rotate(q2, raw_out[:3])
                step_fn = float(np.hypot(np.hypot(rhf[0], rhf[1]), rhf[2] * zw))
            else:
                step_fn = f_norm
            if step_fn > step_pf:
                step_pf = step_fn
            if m_norm > step_pm:
                step_pm = m_norm
            if f_norm > pf:
                pf = f_norm
            if m_norm > pm:
                pm = m_norm

        if do_walk:
            self.walk_current[0] = torch.from_numpy(walk_cur)
            self.k[0] = torch.from_numpy(walk_cur.astype(np.float32))
            self.d[0] = torch.from_numpy((2.0 * damp_ratio * np.sqrt(walk_cur * inertia6)).astype(np.float32))
        self.sim.data.pull()
        self.sim.data.xfrc_applied[0] = torch.from_numpy(d.xfrc_applied.astype(np.float32))
        self.wall_force_z[0] = float(wall_fz)
        self.filtered_sensor[0] = torch.from_numpy(filt.astype(np.float32))
        self.hysteresis_sensor[0] = torch.from_numpy(hyst.astype(np.float32))
        self.wrench[0] = torch.from_numpy(wrench_out.astype(np.float32))
        self.raw_wrench[0] = torch.from_numpy(raw_out.astype(np.float32))
        self.step_peak_force[0] = step_pf
        self.step_peak_moment[0] = step_pm
        self.peak_force[0] = pf
        self.peak_moment[0] = pm

    def snapshot_dict(self, *, include_surface: bool = True) -> dict:
        entry = {
            "qpos": self.sim.data.qpos[0].tolist(),
            "grasp_offset": (self.offset[0] - self.nominal_offset[0]).tolist(),
            "calib_offset": self.calibration[0].tolist(),
            "goal": self.goal[0].tolist(),
            "wrench": self.wrench[0].tolist(),
            "stiffness": self.k[0].tolist(),
            "damping": self.d[0].tolist(),
            "episode_step": int(self.length[0]),
            "clearance": self.clearance,
            "difficulty": self.difficulty,
        }
        if include_surface and self.surf_slice is not None:
            entry["surface_geom_pos"] = self.sim.model.geom_pos[0, self.surf_slice].tolist()
            entry["surface_geom_quat"] = self.sim.model.geom_quat[0, self.surf_slice].tolist()
        return entry


def run_interactive(args: argparse.Namespace) -> None:
    from evaluate import load_checkpoint, REASONS

    torch.set_num_threads(1)
    ckpt_path = args.run_dir / "checkpoint.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_iter = int(saved["iter"])
    metadata, actor = load_checkpoint(ckpt_path, "cpu")
    base_cfg = InsertionCfg.from_checkpoint(metadata["environment"])

    server = viser.ViserServer(host=args.host, port=args.port, label="Interactive Real-Time Insertion")

    @server.on_client_connect
    def configure_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.10, 0.12, 0.08)
        client.camera.look_at = (0., 0., -0.005)
        client.camera.up_direction = (0., 0., 1.)

    status_md = server.gui.add_markdown("正在初始化实时交互环境...")

    with server.gui.add_folder("0. 运行控制 (Controls)", expand_by_default=True):
        btn_start = server.gui.add_button("▶ 开始实时插入 (Start)", color="green")
        btn_pause = server.gui.add_button("⏸ 暂停 / 继续 (Pause)", disabled=True)
        btn_reset = server.gui.add_button("⏹ 停止并重置初态 (Reset)", disabled=True)
        gui_stochastic = server.gui.add_checkbox(
            "🎲 按策略 std 随机采样动作 (Stochastic)", initial_value=False)
        btn_reload = server.gui.add_button("🔄 重载最新训练 Checkpoint")

    with server.gui.add_folder("1. 插头、初始位姿与凹凸地形 (开始后锁定)", expand_by_default=True):
        gui_shape = server.gui.add_dropdown(
            "插头形状 (Geometry)", options=ALL_SHAPES,
            initial_value=args.shape or "square")
        gui_x = server.gui.add_slider("初始 X (mm)", min=-25.0, max=25.0, step=0.5, initial_value=12.0)
        gui_y = server.gui.add_slider("初始 Y (mm)", min=-25.0, max=25.0, step=0.5, initial_value=-10.0)
        gui_z = server.gui.add_slider("初始 Z 高度 (mm)", min=1.0, max=25.0, step=0.5, initial_value=10.0)
        gui_rx = server.gui.add_slider("初始倾角 Rx (deg)", min=-15.0, max=15.0, step=0.5, initial_value=3.0)
        gui_ry = server.gui.add_slider("初始倾角 Ry (deg)", min=-15.0, max=15.0, step=0.5, initial_value=-3.0)
        gui_rz = server.gui.add_slider("初始偏航 Rz (deg)", min=-25.0, max=25.0, step=0.5, initial_value=5.0)
        max_depth_mm = round(base_cfg.hole_depth_m * base_cfg.target_depth_max_fraction * 1000, 1)
        min_depth_mm = round(base_cfg.target_depth_min_m * 1000, 1)
        gui_depth = server.gui.add_slider(
            "目标深度 (mm)", min=min_depth_mm, max=max_depth_mm, step=0.5, initial_value=15.0)
        btn_random_pose = server.gui.add_button("🎲 随机生成初始位姿")
        gui_rough_amp = server.gui.add_slider(
            "凹凸地形峰谷幅度 (mm, 0=平坦)", min=0.0, max=4.0, step=0.25,
            initial_value=round(base_cfg.rough_max_amplitude_m * 1000, 2)
            if base_cfg.rough_surface else 0.0)
        gui_rough_wl_min = server.gui.add_slider(
            "地形最小空间波长 (mm)", min=5.0, max=25.0, step=1.0,
            initial_value=round(base_cfg.rough_wavelength_min_m * 1000, 1))
        gui_rough_wl_max = server.gui.add_slider(
            "地形最大空间波长 (mm)", min=10.0, max=50.0, step=1.0,
            initial_value=round(base_cfg.rough_wavelength_max_m * 1000, 1))
        btn_random_terrain = server.gui.add_button("🏔️ 随机刷新凹凸地形图案")

    with server.gui.add_folder("2. 观测噪声与偏置滑条 (实时可调)", expand_by_default=True):
        gui_noise_scale = server.gui.add_slider(
            "总噪声倍率 (Noise Scale)", min=0.0, max=2.0, step=0.05,
            initial_value=0.0 if args.zero_noise else 1.0)
        gui_pos_bias = server.gui.add_slider(
            "位置常值偏置 (mm)", min=0.0, max=5.0, step=0.1,
            initial_value=round(base_cfg.position_bias * 1000, 2))
        gui_grasp_xy_err = server.gui.add_slider(
            "横向支点偏差 σ_xy (mm)", min=0.0, max=6.0, step=0.25,
            initial_value=round(base_cfg.grasp_position_error * 1000, 2))
        default_grasp_z = (
            base_cfg.grasp_position_error if base_cfg.grasp_z_error is None else base_cfg.grasp_z_error)
        gui_grasp_z_err = server.gui.add_slider(
            "纵向支点偏差 σ_z (mm)", min=0.0, max=10.0, step=0.25,
            initial_value=round(default_grasp_z * 1000, 2))
        gui_calib_err = server.gui.add_slider(
            "尖端标定误差 (mm)", min=0.0, max=5.0, step=0.1,
            initial_value=round(base_cfg.tip_calibration_error * 1000, 2))
        gui_rot_bias = server.gui.add_slider(
            "姿态常值零偏 (deg)", min=0.0, max=5.0, step=0.1,
            initial_value=round(math.degrees(base_cfg.rotation_bias), 2))
        gui_pos_noise = server.gui.add_slider(
            "位置逐步白噪声 (mm)", min=0.0, max=2.0, step=0.05,
            initial_value=round(base_cfg.position_noise * 1000, 2))
        gui_rot_noise = server.gui.add_slider(
            "姿态逐步白噪声 (deg)", min=0.0, max=5.0, step=0.1,
            initial_value=round(math.degrees(base_cfg.rotation_noise), 2))
        gui_force_bias = server.gui.add_slider(
            "力常值零偏 (N)", min=0.0, max=3.0, step=0.05,
            initial_value=round(base_cfg.force_bias, 2))
        gui_force_noise = server.gui.add_slider(
            "力逐步白噪声 (N)", min=0.0, max=1.0, step=0.05,
            initial_value=round(base_cfg.force_noise, 2))
        gui_moment_bias = server.gui.add_slider(
            "力矩常值零偏 (N·m)", min=0.0, max=0.30, step=0.01,
            initial_value=round(base_cfg.moment_bias, 3))
        gui_moment_noise = server.gui.add_slider(
            "力矩逐步白噪声 (N·m)", min=0.0, max=0.05, step=0.001,
            initial_value=round(base_cfg.moment_noise, 3))
        btn_resample_bias = server.gui.add_button("🎲 重新采样扰动方向/刚度主轴/延迟")
        gui_auto_resample = server.gui.add_checkbox(
            "重置/随机位姿时自动重采样本局扰动实例", initial_value=True)

    with server.gui.add_folder("3. 物理刚度与摩擦滑条 (实时可调)", expand_by_default=False):
        gui_k_xyz = server.gui.add_slider(
            "平移刚度 K_xyz (N/m)", min=100.0, max=2500.0, step=10.0, initial_value=380.0)
        gui_k_rot = server.gui.add_slider(
            "旋转刚度 K_rot (N·m/rad)", min=1.5, max=12.0, step=0.5, initial_value=4.0)
        gui_friction = server.gui.add_slider(
            "接触摩擦系数 μ", min=0.10, max=1.50, step=0.05, initial_value=0.70)
        gui_stiction_f = server.gui.add_slider(
            "平移静摩擦死区 (N)", min=0.0, max=1.20, step=0.05,
            initial_value=round(base_cfg.stiction_force, 2))
        gui_stiction_m = server.gui.add_slider(
            "旋转静摩擦死区 (N·m)", min=0.0, max=0.12, step=0.005,
            initial_value=round(base_cfg.stiction_moment, 3))
        gui_wall_f = server.gui.add_slider(
            "孔壁轴向摩擦力 (N)", min=0.0, max=8.0, step=0.2,
            initial_value=round(base_cfg.wall_friction_force, 2))
        gui_term_overload = server.gui.add_checkbox(
            "力/力矩超限立即终止", initial_value=False)

    pose_controls = (
        gui_shape, gui_x, gui_y, gui_z, gui_rx, gui_ry, gui_rz, gui_depth, btn_random_pose,
        gui_rough_amp, gui_rough_wl_min, gui_rough_wl_max, btn_random_terrain,
    )
    lock = threading.Lock()
    state = {
        "mode": "setup",  # "setup", "running", "paused", "finished"
        "finish_reason": "",
        "last_action": [0.0] * 6,
        "last_std": [0.0] * 6,
        "dirty_pose": True,
        "dirty_params": True,
        "start_requested": False,
        "resample_requested": False,
        "resample_terrain_requested": False,
        "reload_requested": False,
    }

    interactive_scene_cfg = replace(base_cfg, rough_surface=True)
    plant = RealtimePlant(gui_shape.value, interactive_scene_cfg)
    scene = SnapshotScene(server, gui_shape.value, interactive_scene_cfg)

    def set_pose_locked(locked: bool) -> None:
        for ctrl in pose_controls:
            ctrl.disabled = locked

    def sync_plant_parameters(recompute_geom: bool) -> None:
        plant.apply_parameters(
            noise_scale=gui_noise_scale.value,
            pos_bias_mm=gui_pos_bias.value,
            grasp_xy_err_mm=gui_grasp_xy_err.value,
            grasp_z_err_mm=gui_grasp_z_err.value,
            calib_err_mm=gui_calib_err.value,
            rot_bias_deg=gui_rot_bias.value,
            pos_noise_mm=gui_pos_noise.value,
            rot_noise_deg=gui_rot_noise.value,
            force_bias_n=gui_force_bias.value,
            force_noise_n=gui_force_noise.value,
            moment_bias_nm=gui_moment_bias.value,
            moment_noise_nm=gui_moment_noise.value,
            k_xyz=gui_k_xyz.value,
            k_rot=gui_k_rot.value,
            friction_mu=gui_friction.value,
            stiction_f=gui_stiction_f.value,
            stiction_m=gui_stiction_m.value,
            wall_f=gui_wall_f.value,
            terminate_on_overload=gui_term_overload.value,
            recompute_geom=recompute_geom,
        )

    def sync_setup_pose() -> None:
        nonlocal plant, scene
        if gui_shape.value != plant.shape:
            scene.remove()
            plant = RealtimePlant(gui_shape.value, interactive_scene_cfg)
            scene = SnapshotScene(server, gui_shape.value, interactive_scene_cfg)
        plant.apply_terrain(gui_rough_amp.value, gui_rough_wl_min.value, gui_rough_wl_max.value)
        sync_plant_parameters(recompute_geom=True)
        plant.set_initial_pose(
            gui_x.value, gui_y.value, gui_z.value,
            gui_rx.value, gui_ry.value, gui_rz.value,
            gui_depth.value,
        )
        scene.update(plant.snapshot_dict())

    for ctrl in (gui_shape, gui_x, gui_y, gui_z, gui_rx, gui_ry, gui_rz, gui_depth,
                 gui_rough_amp, gui_rough_wl_min, gui_rough_wl_max):
        @ctrl.on_update
        def _on_pose_change(event) -> None:
            del event
            with lock:
                if state["mode"] == "setup":
                    state["dirty_pose"] = True

    for ctrl in (gui_noise_scale, gui_pos_bias, gui_grasp_xy_err, gui_grasp_z_err,
                 gui_calib_err, gui_rot_bias,
                 gui_pos_noise, gui_rot_noise, gui_force_bias, gui_force_noise,
                 gui_moment_bias, gui_moment_noise, gui_k_xyz, gui_k_rot,
                 gui_friction, gui_stiction_f, gui_stiction_m, gui_wall_f, gui_term_overload):
        @ctrl.on_update
        def _on_param_change(event) -> None:
            del event
            with lock:
                state["dirty_params"] = True
                if state["mode"] == "setup":
                    state["dirty_pose"] = True

    @btn_random_pose.on_click
    def _on_random_pose(event) -> None:
        del event
        with lock:
            if state["mode"] != "setup":
                return
            rng = np.random.default_rng()
            gui_x.value = round(float(rng.uniform(-20.0, 20.0)) * 2) / 2
            gui_y.value = round(float(rng.uniform(-20.0, 20.0)) * 2) / 2
            z_floor = max(4.0, float(gui_rough_amp.value) + 3.5)
            gui_z.value = round(float(rng.uniform(z_floor, 20.0)) * 2) / 2
            gui_rx.value = round(float(rng.uniform(-8.0, 8.0)) * 2) / 2
            gui_ry.value = round(float(rng.uniform(-8.0, 8.0)) * 2) / 2
            gui_rz.value = round(float(rng.uniform(-15.0, 15.0)) * 2) / 2
            gui_depth.value = round(float(rng.uniform(min_depth_mm, max_depth_mm)) * 2) / 2
            if gui_auto_resample.value:
                state["resample_requested"] = True
            state["dirty_pose"] = True

    @btn_random_terrain.on_click
    def _on_random_terrain(event) -> None:
        del event
        with lock:
            if state["mode"] != "setup":
                return
            state["resample_terrain_requested"] = True
            state["dirty_pose"] = True

    @btn_resample_bias.on_click
    def _on_resample_bias(event) -> None:
        del event
        with lock:
            state["resample_requested"] = True
            state["dirty_params"] = True
            if state["mode"] == "setup":
                state["dirty_pose"] = True

    @btn_start.on_click
    def _on_start(event) -> None:
        del event
        with lock:
            if state["mode"] == "running":
                return
            set_pose_locked(True)
            btn_start.disabled = True
            btn_pause.disabled = False
            btn_reset.disabled = False
            state["start_requested"] = True

    @btn_pause.on_click
    def _on_pause(event) -> None:
        del event
        with lock:
            if state["mode"] == "running":
                state["mode"] = "paused"
            elif state["mode"] == "paused":
                state["mode"] = "running"

    @btn_reset.on_click
    def _on_reset(event) -> None:
        del event
        with lock:
            state["mode"] = "setup"
            state["finish_reason"] = ""
            state["last_action"] = [0.0] * 6
            state["last_std"] = [0.0] * 6
            state["start_requested"] = False
            set_pose_locked(False)
            btn_start.disabled = False
            btn_pause.disabled = True
            btn_reset.disabled = True
            if gui_auto_resample.value:
                state["resample_requested"] = True
            state["dirty_pose"] = True

    @btn_reload.on_click
    def _on_reload(event) -> None:
        del event
        with lock:
            state["reload_requested"] = True

    def render_status() -> None:
        pos, q, _, _ = plant.state()
        t_pos = tip_pose(pos, q, plant.offset)[0] * 1000.0
        lowest_z = float(t_pos[2] + plant.minimum_peg_height_offset(q)[0] * 1000.0)
        tilt, yaw = plant.alignment(q)
        w = plant.wrench[0].tolist()
        f_norm = float(torch.linalg.vector_norm(plant.wrench[0, :3]))
        m_norm = float(torch.linalg.vector_norm(plant.wrench[0, 3:]))
        a = state["last_action"]
        s_std = state["last_std"]
        policy_mode = "随机采样 (mean + std·ε)" if gui_stochastic.value else "确定性均值 (mean)"
        mode_labels = {
            "setup": "🛠 **准备就绪**（请调节初始位置/姿态与插头，点击 **▶ 开始实时插入**）",
            "running": "🚀 **实时插入运行中 (50 Hz)**（插头与初始位姿已锁定）",
            "paused": "⏸ **已暂停**（点击 **⏸ 暂停 / 继续** 恢复，或点击 **⏹ 停止并重置**）",
            "finished": f"🏁 **回合结束：{state['finish_reason']}**（点击 **⏹ 停止并重置** 解锁修改）",
        }
        grasp_mm = ((plant.offset[0] - plant.nominal_offset[0]) * 1000.0).tolist()
        bias_mm = (plant.bias[0] * 1000.0).tolist()
        rb_deg = (quat_rotvec(plant.rotation_bias_q)[0] * (180.0 / math.pi)).tolist()
        act_d = int(plant.delay[0])
        obs_d = int(plant.position_delay_frames[0]) if plant.cfg.position_observation_delay else 0
        status_md.content = (
            f"{mode_labels[state['mode']]}  \n"
            f"**模型轮次 (Iteration)**: `{ckpt_iter}` | **形状**: `{plant.shape}` | "
            f"**地形峰值**: `{float(plant.surface_max_height[0])*1000:.2f} mm` | "
            f"**动作模式**: `{policy_mode}`  \n"
            f"**本局扰动实例**: 支点 `[{grasp_mm[0]:+5.2f}, {grasp_mm[1]:+5.2f}, {grasp_mm[2]:+5.2f}] mm` | "
            f"目标偏置 `[{bias_mm[0]:+5.2f}, {bias_mm[1]:+5.2f}, {bias_mm[2]:+5.2f}] mm` | "
            f"姿态零偏 `[{rb_deg[0]:+4.2f}, {rb_deg[1]:+4.2f}, {rb_deg[2]:+4.2f}]°` | "
            f"延迟 `act={act_d}/obs={obs_d}帧`  \n"
            f"**当前步数**: `{int(plant.length[0])}` (`{int(plant.length[0])*0.02:.2f} s`) | "
            f"**达标保持**: `{int(plant.hold[0])}/{plant.cfg.success_hold_steps}`  \n"
            f"**真实尖端 XYZ [mm]**: `[{t_pos[0]:+6.2f}, {t_pos[1]:+6.2f}, {t_pos[2]:+6.2f}]` "
            f"(最低点 `{lowest_z:+6.2f}` / 目标 `{-gui_depth.value:+6.2f}`)  \n"
            f"**姿态误差**: 倾角 `{math.degrees(float(tilt[0])):5.2f}°` | "
            f"对称折叠偏航 `{math.degrees(float(yaw[0])):5.2f}°`  \n"
            f"**滤波尖端力 [N]**: `[{w[0]:+5.2f}, {w[1]:+5.2f}, {w[2]:+5.2f}]` "
            f"(`|F|={f_norm:5.2f} N`, 峰值 `{float(plant.peak_force[0]):5.2f} N`)  \n"
            f"**滤波尖端力矩 [N·m]**: `[{w[3]:+5.3f}, {w[4]:+5.3f}, {w[5]:+5.3f}]` "
            f"(`|M|={m_norm:5.3f} N·m`, 峰值 `{float(plant.peak_moment[0]):5.3f} N·m`)  \n"
            f"**实时策略动作**: `dp=[{a[0]:+5.2f}, {a[1]:+5.2f}, {a[2]:+5.2f}] mm` | "
            f"`dr=[{a[3]*0.5:+5.2f}, {a[4]*0.5:+5.2f}, {a[5]*0.5:+5.2f}]°`  \n"
            f"**策略探索 std**: `[{s_std[0]:.2f}, {s_std[1]:.2f}, {s_std[2]:.2f} mm | "
            f"{s_std[3]*0.5:.2f}, {s_std[4]*0.5:.2f}, {s_std[5]*0.5:.2f}°]`"
        )

    frames = 0
    try:
        with torch.no_grad():
            while args.frames == 0 or frames < args.frames:
                started = time.monotonic()
                with lock:
                    if state["reload_requested"]:
                        s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                        ckpt_iter = int(s["iter"])
                        metadata, actor = load_checkpoint(ckpt_path, "cpu")
                        actor.reset()
                        state["reload_requested"] = False
                        render_status()
                    if state["resample_requested"]:
                        plant.resample_fixed_directions()
                        state["resample_requested"] = False
                    if state["resample_terrain_requested"]:
                        plant.resample_terrain_waves()
                        state["resample_terrain_requested"] = False
                    if state["start_requested"]:
                        sync_setup_pose()
                        actor.reset()
                        state["last_action"] = [0.0] * 6
                        state["last_std"] = [0.0] * 6
                        state["finish_reason"] = ""
                        state["mode"] = "running"
                        state["dirty_pose"] = False
                        state["dirty_params"] = False
                        state["start_requested"] = False
                    if state["mode"] == "setup":
                        if state["dirty_pose"]:
                            sync_setup_pose()
                            state["dirty_pose"] = False
                            state["dirty_params"] = False
                            render_status()
                    elif state["mode"] == "running":
                        if state["dirty_params"]:
                            sync_plant_parameters(recompute_geom=False)
                            state["dirty_params"] = False
                        obs_actor, _ = plant.observations()
                        sampled = actor(TensorDict({"actor": obs_actor}, [1]), stochastic_output=True)
                        action = sampled if gui_stochastic.value else actor.distribution.mean.clone()
                        state["last_action"] = action[0].tolist()
                        state["last_std"] = actor.distribution.std[0].tolist()
                        _, done, info = plant.step(action)
                        scene.update(plant.snapshot_dict(include_surface=False))
                        reason_code = int(info["reason"][0])
                        if bool(done[0]) and reason_code != 4:
                            reason_map = {
                                1: "✅ 插入成功 (Success)",
                                2: "⚠️ 力/力矩超限 (Wrench Overload)",
                                3: "⚠️ 逃出工作空间或倾角超限 (Escaped Workspace)",
                            }
                            state["finish_reason"] = reason_map.get(reason_code, REASONS.get(reason_code, "done"))
                            state["mode"] = "finished"
                            btn_pause.disabled = True
                            render_status()
                        elif frames % 4 == 0:
                            render_status()
                    elif state["mode"] in ("paused", "finished"):
                        if state["dirty_params"]:
                            sync_plant_parameters(recompute_geom=False)
                            state["dirty_params"] = False
                        if frames % 10 == 0:
                            render_status()
                frames += 1
                time.sleep(max(0.003, 0.02 - (time.monotonic() - started)))
    finally:
        server.stop()


class EnvScene:
    """按 GPU 仿真第 0 个世界的实际碰撞体渲染，包含逐回合形状参数与凸台"""

    def __init__(self, server: viser.ViserServer | Any, group: Any):
        self.server, self.group, self.model = server, group, group.model
        tool = self.model.body("tool").id
        self.peg_ids = set(np.flatnonzero(self.model.geom_bodyid == tool).tolist())
        self.rim_ids = set(group.rim_geom_ids)
        self.root = server.scene.add_frame(f"/env/{uuid4().hex}", show_axes=False)
        self.handles: dict[int, Any] = {}
        self.tip = server.scene.add_frame(f"{self.root.name}/tip", axes_length=0.006, axes_radius=0.0002)
        self.goal = server.scene.add_frame(f"{self.root.name}/goal", axes_length=0.006, axes_radius=0.0002)
        self.prior = server.scene.add_frame(f"{self.root.name}/prior", axes_length=0.004, axes_radius=0.0002)
        self.wrench_frame = server.scene.add_frame(f"{self.root.name}/wrench", show_axes=False)
        self.wrench_vectors = [server.scene.add_line_segments(
            f"{self.wrench_frame.name}/{name}", vector_arrow(np.zeros(3)), color,
            thickness=6., thickness_units="screen", position=(0., offset, 0.))
            for name, color, offset in (("force", (255, 80, 60), 0.008), ("torque", (80, 140, 255), -0.008))]

    def rebuild(self, frame: dict) -> None:
        # 碰撞体尺寸逐回合改写，每回合按当前尺寸重建网格
        size = self.group.sim.model.geom_size[0].detach().cpu().numpy().astype(np.float64)
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mjtGeom.mjGEOM_BOX:
                mesh = trimesh.creation.box(extents=2*size[i])
            else:
                mesh = geom_mesh(self.model, i)
            rgba = self.model.geom_rgba[i]
            opacity = 1.0 if i in self.peg_ids else 0.95 if i in self.rim_ids else 0.4
            self.handles[i] = self.server.scene.add_mesh_simple(
                f"{self.root.name}/geom_{i}", mesh.vertices, mesh.faces,
                color=tuple(int(c*255) for c in rgba[:3]), opacity=opacity, side="double")
        self.prior.position = frame["prior"]
        self.goal.position = frame["goal"]
        self.update(frame, static=True)

    def update(self, frame: dict, static: bool = False) -> None:
        with self.server.atomic():
            for i, handle in self.handles.items():
                if not static and i not in self.peg_ids:
                    continue
                position = frame["xpos"][i]
                # 停放到远处的未启用碰撞体不显示
                handle.visible = bool(np.linalg.norm(position) < 0.5)
                q = np.empty(4)
                mju_mat2Quat(q, np.ascontiguousarray(frame["xmat"][i].reshape(-1)))
                handle.position = position
                handle.wxyz = q
            self.tip.position, self.tip.wxyz = frame["tip"], frame["q"]
            self.wrench_frame.position = frame["tip"] + np.array([0.03, 0., 0.02])
            self.wrench_frame.wxyz = frame["q"]
            for handle, value, reference in zip(self.wrench_vectors, (frame["wrench"][:3], frame["wrench"][3:]),
                                                (5., 0.1), strict=True):
                vector = visible_wrench_vector(value, reference)
                length = np.linalg.norm(vector)
                # 撞击力可达数百牛，箭头长度截断在 6 cm
                handle.points = vector_arrow(vector*min(1., 0.06/max(length, 1e-12)))

    def remove(self) -> None:
        self.root.remove()


def search_map_html(env: Any, scale: int = 6) -> str:
    """环境 0 的孔位后验概率图：按当前最大值归一的对数色标，白点=尖端，红点=真实孔位；X 向右、Y 向上"""
    from PIL import Image

    m = env.search_map
    logits = (m.log_prior[0] + m.erased[0]).float()
    # 每帧按当前后验整体重新着色，擦除后剩余格的概率上升会同步变亮
    v = ((logits - logits.max()) / 8. + 1.).clamp(0., 1.).cpu().numpy()
    rgb = np.stack((v, np.sqrt(v) * 0.85, 0.6 * (1. - v)), -1)
    image = np.kron(np.flipud(rgb.transpose(1, 0, 2)), np.ones((scale, scale, 1)))
    truth = env.groups[0].search_map_target()[0].to(m.log_prior.device)
    points = [] if m.position is None else [(m.position[0].tolist(), (1., 1., 1.))]
    points.append(((truth / CELL_M + GRID / 2).tolist(), (1., 0., 0.)))
    # 按连续格坐标画 2×2 像素小点，不遮挡格子本身的概率颜色
    size = GRID * scale
    for (x, y), color in points:
        row = min(max(round((GRID - y) * scale), 1), size - 1)
        col = min(max(round(x * scale), 1), size - 1)
        image[row - 1:row + 1, col - 1:col + 1] = color
    image = (image * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    # 固定定位到窗口左上角，不占右侧控制面板
    return (f'<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(20,20,20,0.85);'
            f'padding:6px;border-radius:6px;color:#eee;font-size:12px;line-height:1.4">'
            f'孔位概率图（X→ Y↑，{CELL_M*1000:.0f} mm/格，{GRID}×{GRID}）<br>'
            f'<img src="data:image/png;base64,{encoded}" style="display:block;image-rendering:pixelated"><br>'
            f'亮=后验概率高（相对当前最大值，对数色标 8 nats） <span style="color:#fff">■</span>尖端 '
            f'<span style="color:#f33">■</span>真实孔位</div>')


def run_env_interactive(args: argparse.Namespace) -> None:
    """GPU 单环境逐步实时仿真：物理、观测与训练完全一致，回合参数可在 GUI 中覆盖"""
    from evaluate import load_checkpoint, REASONS
    from insertion import InsertionEnv
    from insertion.env import LOWPASS_OFF_MAX_RAD_S
    from insertion.shapes.base import ScaledPolygonShape

    torch.set_num_threads(1)
    device = args.device
    ckpt_path = args.run_dir / "checkpoint.pt"

    def load_policy():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        metadata, actor = load_checkpoint(ckpt_path, device)
        return int(saved["iter"]), metadata, actor

    ckpt_iter, metadata, actor = load_policy()
    base = InsertionCfg.from_checkpoint(metadata["environment"])
    curriculum_state = metadata.get("noise_curriculum_state")
    probe = replay_config(metadata["environment"], base.shapes[0], training_level=True)
    env_cache: dict[str, Any] = {"env": None, "key": None, "scene": None}

    def build_env(shape: str, terminate: bool):
        key = (shape, terminate)
        if env_cache["key"] != key:
            if env_cache["scene"] is not None:
                env_cache["scene"].remove()
            cfg = replace(replay_config(metadata["environment"], shape, training_level=True),
                          terminate_on_overload=terminate, enable_timeout=False)
            env = InsertionEnv(cfg, device)
            if env.noise_curriculum is not None and curriculum_state is not None:
                env.noise_curriculum.load_levels(curriculum_state)
            env_cache.update(env=env, key=key, scene=EnvScene(server, env.groups[0]))
        return env_cache["env"], env_cache["scene"]

    server = viser.ViserServer(host=args.host, port=args.port, label="Live GPU insertion")

    @server.on_client_connect
    def configure_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.05, -0.07, 0.06)
        client.camera.look_at = (0., 0., -0.003)
        client.camera.up_direction = (0., 0., 1.)

    status_md = server.gui.add_markdown("正在构建 GPU 仿真环境...")
    map_panel = server.gui.add_html("") if base.search_map else None
    # 课程默认值取检查点当前等级
    probe_env, _ = build_env(args.shape if args.shape in base.shapes else
                             ("dual" if "dual" in base.shapes else base.shapes[0]), False)
    curriculum = probe_env.noise_curriculum
    default_clearance = (curriculum.clearance_for(probe_env.groups[0].shape) if curriculum is not None
                         else base.clearance)*1000
    default_difficulty = curriculum.difficulty if curriculum is not None else 1.
    curriculum_noise = None if curriculum is None else curriculum.scale_for(probe_env.groups[0].shape)
    default_noise = probe_env.groups[0].noise_scale if curriculum_noise is None else curriculum_noise
    mm = lambda value: round(value*1000/base.scale, 2)

    with server.gui.add_folder("0. 运行控制", expand_by_default=True):
        btn_run = server.gui.add_button("▶ 开始 / 继续", color="green")
        btn_pause = server.gui.add_button("⏸ 暂停")
        btn_step = server.gui.add_button("⏭ 单步（暂停时）")
        btn_new = server.gui.add_button("🔄 新回合（按下方设置重新采样）")
        gui_speed = server.gui.add_slider("播放速度倍率", min=0.05, max=2.0, step=0.05, initial_value=1.0)
        gui_stochastic = server.gui.add_checkbox("按策略 std 随机采样动作", initial_value=False)
        gui_auto = server.gui.add_checkbox("回合结束后自动开新回合", initial_value=False)
        btn_reload = server.gui.add_button("重载最新 checkpoint")
    with server.gui.add_folder("1. 课程与终止（新回合生效）", expand_by_default=True):
        gui_shape = server.gui.add_dropdown("形状", options=base.shapes,
                                            initial_value=env_cache["key"][0])
        gui_clearance = server.gui.add_slider("间隙 (mm)", min=0.05, max=2.0, step=0.05,
                                              initial_value=round(default_clearance, 2))
        gui_difficulty = server.gui.add_slider("难度", min=0., max=1., step=0.05,
                                               initial_value=round(default_difficulty, 2))
        gui_noise = server.gui.add_slider("噪声倍率（实时）", min=0., max=1.5, step=0.05,
                                          initial_value=round(default_noise, 2))
        gui_overload = server.gui.add_checkbox("超限立即终止（评估口径，训练不终止）", initial_value=False)
    with server.gui.add_folder("2. 凸台与形状参数（新回合生效）", expand_by_default=True):
        gui_rim = server.gui.add_dropdown("凸台", options=("随机", "无凸台", "有凸台"), initial_value="随机")
        gui_rim_h = server.gui.add_slider("凸台高度 (mm)", min=mm(probe.rim_height_min_m),
                                          max=mm(probe.rim_height_max_m), step=0.1,
                                          initial_value=mm(probe.rim_height_max_m))
        gui_rim_w = server.gui.add_slider("凸台宽度 (mm)", min=mm(probe.rim_width_min_m),
                                          max=mm(probe.rim_width_max_m), step=0.1,
                                          initial_value=mm(probe.rim_width_min_m))
        gui_shape_random = server.gui.add_checkbox("形状参数随机", initial_value=True)
        gui_params = [server.gui.add_slider(label, min=0.8, max=1.2, step=0.01, initial_value=1.0)
                      for label in ("参数0倍率（dual 插脚厚）", "参数1倍率（dual 插脚宽）",
                                    "参数2倍率（dual 槽距 p）")]
    with server.gui.add_folder("3. 初始位姿（新回合生效）", expand_by_default=True):
        gui_pose_random = server.gui.add_checkbox("初始位姿随机", initial_value=True)
        xy_max = mm(probe.initial_xy)
        gui_x = server.gui.add_slider("初始 X (mm)", min=-xy_max, max=xy_max, step=0.25, initial_value=0.)
        gui_y = server.gui.add_slider("初始 Y (mm)", min=-xy_max, max=xy_max, step=0.25, initial_value=0.)
        gui_z = server.gui.add_slider("初始尖端高度 (mm)", min=mm(probe.initial_z_min),
                                      max=mm(probe.initial_z_max), step=0.5, initial_value=15.)
        gui_rx = server.gui.add_slider("初始 Rx (deg)", min=-8., max=8., step=0.5, initial_value=0.)
        gui_ry = server.gui.add_slider("初始 Ry (deg)", min=-8., max=8., step=0.5, initial_value=0.)
        gui_rz = server.gui.add_slider("初始偏航 Rz (deg)", min=-15., max=15., step=0.5, initial_value=0.)
    with server.gui.add_folder("4. 动力学（新回合生效，倍率作用于采样值）", expand_by_default=False):
        gui_k = server.gui.add_slider("平移刚度倍率", min=0.25, max=4., step=0.05, initial_value=1.)
        gui_lowpass = server.gui.add_dropdown("目标低通", options=("随机", "开", "关"), initial_value="随机")
        gui_stiction = server.gui.add_slider("静摩擦死区倍率", min=0., max=2., step=0.05, initial_value=1.)
        gui_wall = server.gui.add_slider("孔壁摩擦倍率", min=0., max=2., step=0.05, initial_value=1.)
        gui_pivot = server.gui.add_slider("上游枢轴倍率", min=0., max=2., step=0.05, initial_value=1.)

    lock = threading.Lock()
    state = {"mode": "paused", "new": True, "step": False, "reload": False}
    for button, key in ((btn_new, "new"), (btn_step, "step"), (btn_reload, "reload")):
        def request(event, key=key) -> None:
            del event
            with lock:
                state[key] = True
        button.on_click(request)

    @btn_run.on_click
    def _on_run(event) -> None:
        del event
        with lock:
            if state["mode"] == "finished":
                state["new"] = True
            state["mode"] = "running"

    @btn_pause.on_click
    def _on_pause(event) -> None:
        del event
        with lock:
            if state["mode"] == "running":
                state["mode"] = "paused"

    ids = torch.arange(1, device=device)

    def override_episode(g: Any) -> None:
        cfg, scale = g.cfg, g.cfg.scale
        definition = g.geometry.definition
        changed = False
        if not gui_shape_random.value and not isinstance(definition, ScaledPolygonShape):
            nominal = torch.tensor(definition.nominal_params(g.geometry), device=device)
            factor = torch.tensor([s.value for s in gui_params[:len(nominal)]], device=device)
            g.shape_params[0, :len(nominal)] = nominal*factor
            changed = True
        if gui_rim.value != "随机" and g.rim_geom_ids:
            g.rim_height[0] = gui_rim_h.value/1000*scale if gui_rim.value == "有凸台" else 0.
            g.rim_width[0] = gui_rim_w.value/1000*scale
            changed = True
        if changed:
            g.apply_shape(ids)
            g.apply_geom_pos(ids, g.offset[ids]-g.nominal_offset[ids])
        if not gui_pose_random.value:
            rv = torch.tensor([[gui_rx.value, gui_ry.value, gui_rz.value]], device=device)*math.pi/180
            q = rotvec_quat(rv)
            tip = torch.tensor([[gui_x.value, gui_y.value, gui_z.value]], device=device)/1000*scale
            tip[:, 2] = tip[:, 2].clamp_min(0.0005*scale - g.minimum_peg_height_offset(q, ids))
            # 与 reset_state 的初始放置一致：位姿、目标、低通与延迟位置观测同步改写
            qpos = g.sim.data.qpos[:]
            qpos[ids, :3] = tip - rotate(q, g.offset[ids])
            qpos[ids, 3:7] = q
            g.tip_ema[ids] = tip
            nominal_tip = tip_pose(qpos[ids, :3], q, g.nominal_offset[ids] + g.calibration[ids])
            g.targets_p[:, ids] = nominal_tip.unsqueeze(0)
            g.targets_q[:, ids] = q.unsqueeze(0)
            g.lowpass_p[ids] = nominal_tip
            g.lowpass_q[ids] = q
            if cfg.position_observation_delay:
                g.reset_position_observation(ids, nominal_tip)
        for tensor in (g.k, g.initial_k, g.walk_lower, g.walk_upper, g.walk_target, g.walk_current):
            tensor[0, :3] *= gui_k.value
        inertia6 = torch.cat((g.apparent_mass[ids].expand(-1, 3), g.inertia[ids]), -1)
        g.d[ids] = 2*torch.sqrt(g.k[ids]*inertia6)*g.damping_ratio[ids]
        if cfg.target_lowpass:
            if gui_lowpass.value != "随机":
                g.lowpass_off[ids] = torch.full_like(g.lowpass_off[ids], gui_lowpass.value == "关")
            off = g.lowpass_off[ids]
            natural = torch.sqrt(g.k[ids, :3]/g.apparent_mass[ids]).clamp_max(LOWPASS_OFF_MAX_RAD_S)
            g.lowpass_omega[ids] = torch.where(off, natural, cfg.target_lowpass_translation_rad_s)
            g.lowpass_zeta[ids] = torch.where(off, g.damping_ratio[ids, :3], 1.)
        g.stiction[0] *= gui_stiction.value
        g.wall_friction[0] *= gui_wall.value
        g.pivot[0] *= gui_pivot.value

    def grab(g: Any) -> dict:
        pos, q, _, _ = g.state()
        tilt, yaw = g.alignment(q)
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        return {
            "xpos": to_np(g.sim.data.geom_xpos[0]), "xmat": to_np(g.sim.data.geom_xmat[0]),
            "tip": to_np(tip_pose(pos, q, g.offset)[0]), "q": to_np(q[0]),
            "goal": to_np(g.goal[0]), "prior": np.r_[to_np(g.bias[0, :2]), 0.],
            "wrench": to_np(g.wrench[0]),
            "peak_force": float(g.peak_force[0]), "peak_moment": float(g.peak_moment[0]),
            "tilt": math.degrees(float(tilt[0])), "yaw": math.degrees(float(yaw[0])),
            "length": int(g.length[0]), "hold": int(g.hold[0]),
        }

    episode: dict[str, Any] = {}
    tally: dict[str, list[int]] = {}

    def episode_info(g: Any) -> str:
        params = (g.shape_params[0]*1000).tolist()
        k = g.k[0].tolist()
        off = bool(g.lowpass_off[0].any()) if g.cfg.target_lowpass else True
        bias = (g.bias[0, :2]*1000).tolist()
        return (f"形状参数 `{', '.join(f'{p:.2f}' for p in params)}` mm | "
                f"凸台 `h={float(g.rim_height[0])*1000:.2f} w={float(g.rim_width[0])*1000:.2f}` mm  \n"
                f"目标偏差 `[{bias[0]:+.2f}, {bias[1]:+.2f}]` mm | "
                f"间隙 `{float(g.shape_params[0, 3])*1000:.2f}` mm | 难度 `{g.difficulty:.2f}`  \n"
                f"K `[{k[0]:.0f}, {k[1]:.0f}, {k[2]:.0f}] N/m` | 低通 `{'关' if off else '开'}` | "
                f"孔壁摩擦 `{float(g.wall_friction[0]):.2f} N` | "
                f"静摩擦 `{float(g.stiction[0, 2]):.2f} N` | "
                f"枢轴 `{float(torch.linalg.vector_norm(g.pivot[0]))*1000:.1f} mm`")

    def render_status(frame: dict) -> None:
        a = episode.get("action", [0.]*6)
        f = frame["wrench"]
        tip_rel = (frame["tip"]-frame["goal"])*1000
        results = " | ".join(f"{s} {v[0]}/{v[1]}" for s, v in tally.items()) or "无"
        labels = {"running": "运行中", "paused": "已暂停", "finished": f"回合结束：**{episode.get('reason', '')}**"}
        status_md.content = (
            f"**{labels[state['mode']]}** | 迭代 `{ckpt_iter}` | 检查点难度 `{default_difficulty:.2f}` | "
            f"终止口径 `{'超限终止' if env_cache['key'][1] else '训练（超限不终止）'}`  \n"
            f"{episode.get('info', '')}  \n"
            f"步数 `{frame['length']}` (`{frame['length']*0.02:.2f} s`) | 保持 `{frame['hold']}`  \n"
            f"尖端相对真孔 `[{tip_rel[0]:+6.2f}, {tip_rel[1]:+6.2f}, {tip_rel[2]:+6.2f}]` mm | "
            f"倾角 `{frame['tilt']:.2f}°` 偏航 `{frame['yaw']:+.2f}°`  \n"
            f"力 `[{f[0]:+6.2f}, {f[1]:+6.2f}, {f[2]:+6.2f}]` N (|F| `{np.linalg.norm(f[:3]):.2f}`) | "
            f"本回合峰值 `{max(frame['peak_force'], episode.get('peak', 0.)):.1f} N`  \n"
            f"动作 dp `[{a[0]:+6.2f}, {a[1]:+6.2f}, {a[2]:+6.2f}]` mm | "
            f"dr `[{a[3]:+5.2f}, {a[4]:+5.2f}, {a[5]:+5.2f}]`  \n"
            f"本次会话结果（成功/回合）: {results}"
        )

    def new_episode() -> tuple[Any, Any, TensorDict]:
        env, scene = build_env(gui_shape.value, gui_overload.value)
        g = env.groups[0]
        g.difficulty = gui_difficulty.value
        g.noise_scale = gui_noise.value
        g.set_clearance(gui_clearance.value/1000*g.cfg.scale)
        with g.warp_on_current_stream():
            g.reset_state(ids)
            override_episode(g)
            g.finish_reset()
        actor.reset()
        if env.search_map is not None:
            env.reset_search_map(torch.ones(1, dtype=torch.bool, device=device))
            actor.search_map = env.search_map
        frame = grab(g)
        scene.rebuild(frame)
        episode.clear()
        episode.update(info=episode_info(g), peak=0.)
        obs = env.get_observations()
        if map_panel is not None:
            map_panel.content = search_map_html(env)
        return env, scene, obs

    frames = 0
    next_step = time.monotonic()
    finished_at = 0.
    try:
        with torch.inference_mode():
            env, scene, obs = new_episode()
            state["new"] = False
            last_frame = grab(env.groups[0])
            render_status(last_frame)
            while args.frames == 0 or frames < args.frames:
                with lock:
                    request = dict(state)
                    state["step"] = state["new"] = state["reload"] = False
                if request["reload"]:
                    ckpt_iter, metadata, actor = load_policy()
                    request["new"] = True
                auto = (request["mode"] == "finished" and gui_auto.value
                        and time.monotonic()-finished_at > 1.)
                if request["new"] or auto:
                    env, scene, obs = new_episode()
                    with lock:
                        if state["mode"] == "finished":
                            state["mode"] = "running" if auto else "paused"
                    last_frame = grab(env.groups[0])
                    render_status(last_frame)
                    next_step = time.monotonic()
                    continue
                now = time.monotonic()
                if not ((request["mode"] == "running" and now >= next_step)
                        or (request["mode"] == "paused" and request["step"])):
                    time.sleep(0.002)
                    continue
                next_step = max(next_step + 0.02/gui_speed.value, now - 0.1)
                g = env.groups[0]
                sampled = actor(obs, stochastic_output=True)
                action = sampled if gui_stochastic.value else actor.distribution.mean.clone()
                episode["action"] = action[0].tolist()
                obs, _, dones, info = env.step(action)
                actor.reset(dones)
                # 课程标量在步内会被环境改回检查点值，按 GUI 恢复
                g.difficulty, g.noise_scale = gui_difficulty.value, gui_noise.value
                frames += 1
                if bool(dones[0]):
                    # 结束步已自动重置，画面停在结束前一步
                    terminal = info["terminal"]
                    code = int(terminal["reason"][0])
                    episode["reason"] = REASONS.get(code, str(code))
                    episode["peak"] = float(terminal["peak_force"][0])
                    record = tally.setdefault(gui_shape.value, [0, 0])
                    record[0] += code == 1
                    record[1] += 1
                    with lock:
                        state["mode"] = "finished"
                    finished_at = time.monotonic()
                    render_status(last_frame)
                    continue
                last_frame = grab(g)
                episode["peak"] = max(episode["peak"], last_frame["peak_force"])
                scene.update(last_frame)
                if frames % 3 == 0 or request["mode"] == "paused":
                    render_status(last_frame)
                    if map_panel is not None:
                        map_panel.content = search_map_html(env)
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--mode", choices=("interactive", "live", "policy"), default="interactive")
    parser.add_argument("--shape", choices=ALL_SHAPES,
                        help="initial geometry; unseen shapes require policy replay or interactive mode")
    parser.add_argument("--device", default="cuda:1", help="independent policy replay device")
    parser.add_argument("--zero-noise", action="store_true",
                        help="disable observation noise and fixed measurement/grasp errors in policy replay")
    parser.add_argument("--ignore-overload", action="store_true",
                        help="simulation replay only: continue past force/moment limits")
    parser.add_argument("--real-profile", action="store_true",
                        help="policy replay with the real2sim evaluation plant and deployment low-pass")
    parser.add_argument("--training-level", action="store_true",
                        help="policy replay at the checkpoint curriculum level (training clearance/difficulty/noise)")
    args = parser.parse_args()
    if args.frames < 0 or not 1 <= args.port <= 65535:
        parser.error("frames must be nonnegative and port must be valid")
    logging.basicConfig(level=logging.INFO)
    if args.mode == "interactive":
        environment = json.loads((args.run_dir / "config.json").read_text())["environment"]
        cfg = InsertionCfg.from_checkpoint(environment)
        # CPU 实时植物不含形状随机化与凸台，这类运行改用 GPU 训练环境逐步仿真
        if cfg.shape_randomization:
            run_env_interactive(args)
        else:
            run_interactive(args)
        return
    replay_only = args.zero_noise or args.ignore_overload or args.real_profile or args.training_level
    if replay_only and args.mode != "policy":
        parser.error("--zero-noise, --ignore-overload, --real-profile and --training-level require --mode policy")
    metadata = json.loads((args.run_dir / "config.json").read_text())
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    if args.shape is not None and args.shape not in cfg.shapes and args.mode != "policy":
        parser.error("unseen --shape requires --mode policy")
    server = viser.ViserServer(host=args.host, port=args.port, label="Insertion training")
    selection = server.gui.add_dropdown("Geometry", options=ALL_SHAPES,
                                       initial_value=args.shape or cfg.shapes[0])
    mode = server.gui.add_dropdown("Source", options=("Training snapshots", "Latest policy replay"),
                                  initial_value=("Latest policy replay" if args.mode == "policy"
                                                 else "Training snapshots"),
                                  disabled=replay_only or selection.value not in cfg.shapes)

    @selection.on_update
    def select_geometry(event) -> None:
        del event
        unseen = selection.value not in cfg.shapes
        if unseen:
            mode.value = "Latest policy replay"
        mode.disabled = replay_only or unseen

    frozen = server.gui.add_checkbox("Freeze view (training continues)", initial_value=False)
    status = server.gui.add_markdown("Waiting for the first training snapshot.")
    server.gui.add_markdown(
        "Snapshots show training environment 0. Policy replay uses deterministic actions, "
        "buffers complete episodes and plays at 1x / 50 Hz, repeating until the next is ready. "
        "Replay does not train."
    )
    if args.zero_noise:
        server.gui.add_markdown(
            "**Zero noise:** measurement/grasp errors off; pose, depth and dynamics remain randomized."
        )
    if args.ignore_overload:
        server.gui.add_markdown(
            "**SIMULATION ONLY: overload termination OFF.** Loads, workspace and timeout checks "
            "remain active. Results do not imply safe motion."
        )

    @server.on_client_connect
    def configure_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.10, 0.12, 0.08)
        client.camera.look_at = (0., 0., -0.005)
        client.camera.up_direction = (0., 0., 1.)

    scene = None
    selected_shape = None
    selected_clearance = None
    frames = 0
    replay = None
    displayed_episode = None
    replay_started = 0.
    last_status_time = 0.
    last_snapshot_key = None
    last_replay_key = None
    previous_tick = time.monotonic()
    try:
        while args.frames == 0 or frames < args.frames:
            started = time.monotonic()
            elapsed_tick = started-previous_tick
            previous_tick = started
            if frozen.value:
                replay_started += elapsed_tick
            path = args.run_dir / "scene.json"
            if path.exists():
                snapshot = json.loads(path.read_text())
                shape = selection.value
                plant = snapshot["plants"].get(shape)
                episode = None
                source = "Training snapshots (2 Hz; training simulation runs slower than real time)"
                if mode.value == "Latest policy replay":
                    if replay is None:
                        replay = PolicyReplay(args.run_dir, args.device, shape, args.zero_noise,
                                              args.ignore_overload, args.real_profile, args.training_level)
                    replay.select(shape)
                    episode = replay.latest()
                    if (frozen.value and displayed_episode is not None
                            and displayed_episode["shape"] == shape):
                        episode = displayed_episode
                    if episode is not None and episode["shape"] == shape:
                        if episode is not displayed_episode:
                            displayed_episode = episode
                            replay_started = time.monotonic()
                            last_replay_key = None
                        index = replay_index(time.monotonic()-replay_started, len(episode["frames"]))
                        plant = episode["frames"][index]
                        source = (f"Replay {args.device}, 1x / 50 Hz | "
                                  f"steps {episode['policy_steps']} | {episode['reason']} | "
                                  f"noise: {'ZERO' if episode['zero_noise'] else 'checkpoint full noise'} | "
                                  f"generation {episode['generation_seconds']:.1f}s")
                    else:
                        episode = None
                        if replay_only or plant is None:
                            status.content = "**Buffering policy episode; no training snapshot displayed.**"
                            frames += 1
                            time.sleep(max(0., 0.02-(time.monotonic()-started)))
                            continue
                        source = "Buffering first policy episode; showing training snapshots meanwhile"
                plant_clearance = float(plant.get("clearance", cfg.clearance))
                if (scene is None or shape != selected_shape
                        or plant_clearance != selected_clearance):
                    if scene is not None:
                        scene.remove()
                    scene = SnapshotScene(server, shape, replace(cfg, clearance=plant_clearance))
                    selected_shape = shape
                    selected_clearance = plant_clearance
                snapshot_key = (shape, snapshot["timestamp"])
                replay_key = (shape, id(episode), index) if episode is not None else None
                changed = ((replay_key != last_replay_key) if episode is not None
                           else (snapshot_key != last_snapshot_key))
                if not frozen.value and changed:
                    scene.update(plant)
                    last_snapshot_key, last_replay_key = snapshot_key, replay_key
                if time.monotonic()-last_status_time < 0.5:
                    frames += 1
                    time.sleep(max(0., 0.02-(time.monotonic()-started)))
                    continue
                last_status_time = time.monotonic()
                heartbeat_path = args.run_dir / "heartbeat.json"
                heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}
                age = max(0., time.time()-snapshot["timestamp"])
                status.content = (
                    f"**{source}**  \n"
                    f"Training snapshot: {'STALE' if age > 10 else 'Live'} | age {age:.1f} s  \n"
                    f"PPO iteration: {heartbeat.get('iteration', 'initial rollout')} | "
                    f"policy steps: {snapshot['policy_steps']}  \n"
                    f"Mean completed-episode reward: {heartbeat.get('mean_episode_reward')}  \n"
                    f"Geometry: **{shape}** | clearance: {plant_clearance*1000:.2f} mm | "
                    f"episode step: {plant['episode_step']} | "
                    f"difficulty: {plant['difficulty']:.3f}  \n"
                    f"Target depth: {-plant['goal'][2]*1000:.2f} mm  \n"
                    f"K [N/m, Nm/rad]: {np.round(plant['stiffness'], 3).tolist()}  \n"
                    f"D [Ns/m, Nms/rad]: {np.round(plant['damping'], 3).tolist()}  \n"
                    f"Fixed grasp offset [mm]: "
                    f"{np.round(np.asarray(plant['grasp_offset'])*1000, 3).tolist()}"
                )
            frames += 1
            time.sleep(max(0., 0.02-(time.monotonic()-started)))
    finally:
        if replay is not None:
            replay.close()
        server.stop()


if __name__ == "__main__":
    main()
