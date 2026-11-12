"""实时训练快照与最新策略回放可视化，不包含任何真机控制通信"""

import argparse
from dataclasses import replace
import json
import logging
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
from insertion.transforms import quat_rotvec, rotate, rotvec_quat, tip_pose


def replay_index(elapsed: float, count: int) -> int:
    if count < 1:
        raise ValueError("replay requires at least one frame")
    return int(max(0., elapsed) / 0.02) % count


def replay_config(environment: dict, shape: str, zero_noise: bool = False,
                  ignore_overload: bool = False) -> InsertionCfg:
    cfg = replace(InsertionCfg.from_checkpoint(environment),
                  shapes=(shape,), shape_env_counts=(), adaptive_sampling=False,
                  envs_per_shape=1, training=False, terminate_on_overload=not ignore_overload)
    if zero_noise:
        cfg = replace(cfg, noise_curriculum=False, noise_curriculum_mode="steps",
                      position_noise=0., rotation_noise=0., force_noise=0., moment_noise=0.,
                      position_bias=0., tip_calibration_error=0., grasp_position_error=0.,
                      sensor_rotation_error=0., force_bias=0., moment_bias=0.)
    return cfg


class PolicyReplay:
    """Generate complete episodes separately, then display them at real-time speed."""

    def __init__(self, run_dir: Path, device: str, shape: str, zero_noise: bool = False,
                 ignore_overload: bool = False):
        self.run_dir, self.device, self.shape = run_dir, device, shape
        self.zero_noise, self.ignore_overload = zero_noise, ignore_overload
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
            import torch
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
                                    self.ignore_overload)
                if env is None or current_cfg != cfg:
                    env = InsertionEnv(cfg, self.device)
                    current_cfg = cfg
                else:
                    env.groups[0].reset(torch.arange(1, device=self.device))
                group = env.groups[0]
                obs = env.get_observations()
                frames = []
                reason = "timeout"
                with torch.inference_mode():
                    for _ in range(env.max_episode_length):
                        if self.stop.is_set() or shape != self.shape:
                            break
                        frames.append({
                            "qpos": group.sim.data.qpos[0].cpu().tolist(),
                            "grasp_offset": (group.offset[0]-group.nominal_offset[0]).cpu().tolist(),
                            "goal": group.goal[0].cpu().tolist(),
                            "wrench": group.wrench[0].cpu().tolist(),
                            "stiffness": group.k[0].cpu().tolist(),
                            "damping": group.d[0].cpu().tolist(),
                            "episode_step": int(group.length[0]),
                            "clearance": group.clearance,
                            "difficulty": group.difficulty,
                        })
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


class SnapshotScene:
    def __init__(self, server: viser.ViserServer | Any, shape: str, cfg: InsertionCfg):
        self.server = server
        self.model = build_model(geometry(shape, cfg.clearance, cfg.scale, cfg.circle_sides,
                                          cfg.plate_outer_radius_m))
        self.data = MjData(self.model)
        self.base_geom_pos = self.model.geom_pos.copy()
        self.tool_id = self.model.body("tool").id
        self.peg_ids = np.flatnonzero(self.model.geom_bodyid == self.tool_id)
        self.nominal_tip = np.array([0., 0., -0.05*cfg.scale])
        # 使用独立随机命名空间，防止切换形状时客户端残留上一几何的四元数姿态
        self.root = server.scene.add_frame(f"/plant/{shape}-{uuid4().hex}", show_axes=False)
        self.handles = []
        for index in range(self.model.ngeom):
            mesh = geom_mesh(self.model, index)
            rgba = self.model.geom_rgba[index]
            name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, index)
            self.handles.append(server.scene.add_mesh_simple(
                f"{self.root.name}/{name}", mesh.vertices, mesh.faces,
                color=(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)),
                opacity=0.45 if index not in self.peg_ids else 1.,
                side="double",
            ))
        self.tip = server.scene.add_frame(f"{self.root.name}/actual_tip", axes_length=0.006, axes_radius=0.0002)
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

    def update(self, plant: dict) -> None:
        wrench = np.asarray(plant["wrench"], dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("viewer wrench must contain six finite components")
        self.data.qpos[:] = plant["qpos"]
        self.model.geom_pos[:] = self.base_geom_pos
        self.model.geom_pos[self.peg_ids] += np.asarray(plant["grasp_offset"])
        mj_forward(self.model, self.data)
        with self.server.atomic():
            for index, handle in enumerate(self.handles):
                q = np.empty(4)
                mju_mat2Quat(q, self.data.geom_xmat[index])
                handle.position = self.data.geom_xpos[index]
                handle.wxyz = q
            rotated = np.empty(3)
            mju_rotVecQuat(rotated, self.nominal_tip+plant["grasp_offset"],
                           self.data.qpos[3:7])
            self.tip.position = self.data.qpos[:3]+rotated
            self.tip.wxyz = self.data.qpos[3:7]
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

    def push(self) -> None:
        self._m.body_mass[:] = self.body_mass[0].numpy()
        self._m.body_inertia[:] = self.body_inertia[0].numpy()
        self._m.geom_friction[:] = self.geom_friction[0].numpy()
        self._m.geom_pos[:] = self.geom_pos[0].numpy()


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


class RealtimePlant(PlantGroup):
    """Single-environment real-time CPU MuJoCo plant with interactive pose/noise/dynamics control."""

    def __init__(self, shape: str, cfg: InsertionCfg):
        cfg = replace(cfg, shapes=(shape,), envs_per_shape=1, shape_env_counts=(),
                      adaptive_sampling=False, training=False, episode_seconds=1e9,
                      physics_cuda_graph=False, observation_cuda_graph=False)
        super().__init__(shape, cfg, "cpu")
        self.grasp_dir = torch.zeros(1, 3)
        self.calib_dir = torch.zeros(1, 3)
        self.pos_bias_dir = torch.zeros(1, 3)
        self.sensor_rot_dir = torch.zeros(1, 3)
        self.force_bias_dir = torch.zeros(1, 3)
        self.moment_bias_dir = torch.zeros(1, 3)
        self.resample_fixed_directions()

    def _build_simulation(self) -> None:
        cfg = self.cfg
        self.geometry = geometry(self.shape, self.clearance, cfg.scale, cfg.circle_sides,
                                 cfg.plate_outer_radius_m)
        self.model = build_model(self.geometry)
        self.model.opt.timestep = cfg.physics_dt
        self.model.opt.gravity[:] = 0
        self.model.opt.iterations = 50
        self.model.opt.ls_iterations = 20
        self.model.bvh_aabb[:, 3:] += 0.01
        self.body_id = self.model.body("tool").id
        self.sim = CpuSim(self.model)
        self.peg_geom_ids = (self.model.geom_bodyid == self.body_id).nonzero()[0].tolist()
        self.default_geom_pos = torch.tensor(
            self.model.geom_pos, device=self.device, dtype=torch.float32)
        self.peg_polygon = torch.tensor(
            self.geometry.peg_polygon, dtype=torch.float32, device=self.device)
        self.hole_polygon = torch.tensor(
            self.geometry.hole_polygon, dtype=torch.float32, device=self.device)

    def resample_fixed_directions(self) -> None:
        self.grasp_dir = torch.randn(1, 3, generator=self.generator)
        self.calib_dir = self.uniform(1, 3, -1., 1.)
        self.pos_bias_dir = self.uniform(1, 3, -1., 1.)
        self.sensor_rot_dir = self.uniform(1, 3, -1., 1.)
        self.force_bias_dir = self.sample_wrench_bias(1, 1.0)
        self.moment_bias_dir = self.sample_wrench_bias(1, 1.0)

    def apply_parameters(self, *, noise_scale: float, pos_bias_mm: float, grasp_err_mm: float,
                         calib_err_mm: float, pos_noise_mm: float, rot_noise_deg: float,
                         force_bias_n: float, force_noise_n: float, moment_bias_nm: float,
                         moment_noise_nm: float, k_xyz: float, k_rot: float, friction_mu: float,
                         stiction_f: float, stiction_m: float, wall_f: float,
                         terminate_on_overload: bool, recompute_geom: bool = True) -> None:
        import math
        self.noise_scale = float(noise_scale)
        self.cfg = replace(
            self.cfg,
            position_noise=pos_noise_mm * 1e-3,
            rotation_noise=math.radians(rot_noise_deg),
            force_noise=force_noise_n,
            moment_noise=moment_noise_nm,
            terminate_on_overload=terminate_on_overload,
        )
        grasp = self.grasp_dir * (grasp_err_mm * 1e-3 * self.noise_scale)
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
        self.sensor_error[0] = rotvec_quat(
            self.sensor_rot_dir * (self.cfg.sensor_rotation_error * self.noise_scale))[0]
        self.wrench_bias[0, :3] = self.force_bias_dir[0] * (force_bias_n * self.noise_scale)
        self.wrench_bias[0, 3:] = self.moment_bias_dir[0] * (moment_bias_nm * self.noise_scale)
        self.k[0] = torch.tensor([k_xyz, k_xyz, k_xyz, k_rot, k_rot, k_rot], dtype=torch.float32)
        self.initial_k[0] = self.k[0].double()
        self.walk_current[0] = self.initial_k[0]
        self.walk_target[0] = self.initial_k[0]
        self.walk_lower[0] = self.initial_k[0]
        self.walk_upper[0] = self.initial_k[0]
        inertia6 = torch.cat((self.mass.expand(1, 3), self.inertia), -1)
        self.d[0] = 2.0 * torch.sqrt(self.k * inertia6)[0]
        self.stiction[0] = torch.tensor([stiction_f]*3 + [stiction_m]*3, dtype=torch.float32)
        self.wall_friction[0] = float(wall_f)

    def set_initial_pose(self, x_mm: float, y_mm: float, z_mm: float,
                         rx_deg: float, ry_deg: float, rz_deg: float,
                         depth_mm: float) -> None:
        import math
        self.sim.reset(torch.zeros(1, dtype=torch.long))
        self.goal[0] = torch.tensor([0., 0., -depth_mm * 1e-3], dtype=torch.float32)
        pos = torch.tensor([[x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3]], dtype=torch.float32)
        rv = torch.tensor([[math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)]],
                          dtype=torch.float32)
        q = rotvec_quat(rv)
        self.sim.data.qpos[0, :3] = (pos - rotate(q, self.offset))[0]
        self.sim.data.qpos[0, 3:7] = q[0]
        self.sim.data.qvel.zero_()
        self.sim.data.xfrc_applied.zero_()
        self.sim.forward()
        nominal_tip = tip_pose(self.sim.data.qpos[:, :3], q,
                               self.nominal_offset + self.calibration)
        self.targets_p[:] = nominal_tip.unsqueeze(0)
        self.targets_q[:] = q.unsqueeze(0)
        self.physics_p_goal[:] = nominal_tip
        self.physics_q_goal[:] = q
        self.delay.zero_()
        for buffer in (self.length, self.hold, self.peak_force, self.peak_moment,
                       self.step_peak_force, self.step_peak_moment,
                       self.last_action, self.wrench, self.raw_wrench, self.filtered_sensor,
                       self.hysteresis_sensor, self.wall_force_z):
            buffer.zero_()
        if self.cfg.position_observation_delay:
            self.reset_position_observation(torch.zeros(1, dtype=torch.long), nominal_tip)

    def snapshot_dict(self) -> dict:
        return {
            "qpos": self.sim.data.qpos[0].tolist(),
            "grasp_offset": (self.offset[0] - self.nominal_offset[0]).tolist(),
            "goal": self.goal[0].tolist(),
            "wrench": self.wrench[0].tolist(),
            "stiffness": self.k[0].tolist(),
            "damping": self.d[0].tolist(),
            "episode_step": int(self.length[0]),
            "clearance": self.clearance,
            "difficulty": self.difficulty,
        }


def run_interactive(args: argparse.Namespace) -> None:
    import math
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

    with server.gui.add_folder("1. 插头与初始位姿 (开始后锁定)", expand_by_default=True):
        gui_shape = server.gui.add_dropdown(
            "插头形状 (Geometry)", options=ALL_SHAPES,
            initial_value=args.shape or "square")
        gui_x = server.gui.add_slider("初始 X (mm)", min=-25.0, max=25.0, step=0.5, initial_value=6.0)
        gui_y = server.gui.add_slider("初始 Y (mm)", min=-25.0, max=25.0, step=0.5, initial_value=-6.0)
        gui_z = server.gui.add_slider("初始 Z 高度 (mm)", min=1.0, max=25.0, step=0.5, initial_value=10.0)
        gui_rx = server.gui.add_slider("初始倾角 Rx (deg)", min=-15.0, max=15.0, step=0.5, initial_value=3.0)
        gui_ry = server.gui.add_slider("初始倾角 Ry (deg)", min=-15.0, max=15.0, step=0.5, initial_value=-3.0)
        gui_rz = server.gui.add_slider("初始偏航 Rz (deg)", min=-25.0, max=25.0, step=0.5, initial_value=5.0)
        gui_depth = server.gui.add_slider("目标深度 (mm)", min=5.0, max=22.5, step=0.5, initial_value=15.0)
        btn_random_pose = server.gui.add_button("🎲 随机生成初始位姿")

    with server.gui.add_folder("2. 观测噪声与偏置滑条 (实时可调)", expand_by_default=True):
        gui_noise_scale = server.gui.add_slider(
            "总噪声倍率 (Noise Scale)", min=0.0, max=2.0, step=0.05,
            initial_value=0.0 if args.zero_noise else 1.0)
        gui_pos_bias = server.gui.add_slider(
            "位置常值偏置 (mm)", min=0.0, max=5.0, step=0.1,
            initial_value=round(base_cfg.position_bias * 1000, 2))
        gui_grasp_err = server.gui.add_slider(
            "夹持偏移误差 (mm)", min=0.0, max=3.0, step=0.05,
            initial_value=round(base_cfg.grasp_position_error * 1000, 2))
        gui_calib_err = server.gui.add_slider(
            "尖端标定误差 (mm)", min=0.0, max=2.0, step=0.05,
            initial_value=round(base_cfg.tip_calibration_error * 1000, 2))
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
        btn_resample_bias = server.gui.add_button("🎲 重新采样偏置方向")

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

    pose_controls = (gui_shape, gui_x, gui_y, gui_z, gui_rx, gui_ry, gui_rz, gui_depth, btn_random_pose)
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
        "reload_requested": False,
    }

    plant = RealtimePlant(gui_shape.value, base_cfg)
    scene = SnapshotScene(server, gui_shape.value, base_cfg)

    def set_pose_locked(locked: bool) -> None:
        for ctrl in pose_controls:
            ctrl.disabled = locked

    def sync_plant_parameters(recompute_geom: bool) -> None:
        plant.apply_parameters(
            noise_scale=gui_noise_scale.value,
            pos_bias_mm=gui_pos_bias.value,
            grasp_err_mm=gui_grasp_err.value,
            calib_err_mm=gui_calib_err.value,
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
            plant = RealtimePlant(gui_shape.value, base_cfg)
            scene = SnapshotScene(server, gui_shape.value, base_cfg)
        sync_plant_parameters(recompute_geom=True)
        plant.set_initial_pose(
            gui_x.value, gui_y.value, gui_z.value,
            gui_rx.value, gui_ry.value, gui_rz.value,
            gui_depth.value,
        )
        scene.update(plant.snapshot_dict())

    for ctrl in (gui_shape, gui_x, gui_y, gui_z, gui_rx, gui_ry, gui_rz, gui_depth):
        @ctrl.on_update
        def _on_pose_change(event) -> None:
            del event
            with lock:
                if state["mode"] == "setup":
                    state["dirty_pose"] = True

    for ctrl in (gui_noise_scale, gui_pos_bias, gui_grasp_err, gui_calib_err,
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
            gui_z.value = round(float(rng.uniform(4.0, 20.0)) * 2) / 2
            gui_rx.value = round(float(rng.uniform(-8.0, 8.0)) * 2) / 2
            gui_ry.value = round(float(rng.uniform(-8.0, 8.0)) * 2) / 2
            gui_rz.value = round(float(rng.uniform(-15.0, 15.0)) * 2) / 2
            gui_depth.value = round(float(rng.uniform(5.0, 22.5)) * 2) / 2
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
        status_md.content = (
            f"{mode_labels[state['mode']]}  \n"
            f"**模型轮次 (Iteration)**: `{ckpt_iter}` | **形状**: `{plant.shape}` | "
            f"**动作模式**: `{policy_mode}`  \n"
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
                        scene.update(plant.snapshot_dict())
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
                    elif state["mode"] in ("paused", "finished"):
                        if state["dirty_params"]:
                            sync_plant_parameters(recompute_geom=False)
                            state["dirty_params"] = False
                        if frames % 10 == 0:
                            render_status()
                frames += 1
                time.sleep(max(0., 0.02 - (time.monotonic() - started)))
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
    args = parser.parse_args()
    if args.frames < 0 or not 1 <= args.port <= 65535:
        parser.error("frames must be nonnegative and port must be valid")
    logging.basicConfig(level=logging.INFO)
    if args.mode == "interactive":
        run_interactive(args)
        return
    replay_only = args.zero_noise or args.ignore_overload
    if replay_only and args.mode != "policy":
        parser.error("--zero-noise and --ignore-overload require --mode policy")
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
                                              args.ignore_overload)
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
