"""基于 mjlab 批量 MuJoCo 仿真的六自由度插孔动力学环境"""

from contextlib import contextmanager
from dataclasses import asdict, replace
import logging
import math
from typing import Any, Iterator

import mujoco_warp as mjwarp
import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .config import ALL_SHAPES, InsertionCfg
from .curriculum import SuccessNoiseCurriculum
from .geometry import build_model, geometry, points_in_polygon, symmetry_count
from .transforms import (action_target, actor_observation, canonical_quat,
                         quat_inv, quat_mul, quat_rotvec, rotate,
                         rotvec_quat, tip_pose, transform_wrench)


@contextmanager
def capture_graph(device: str) -> Iterator[torch.cuda.CUDAGraph]:
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            yield graph
        current.wait_stream(stream)


def bounded_stiffness_step(current: torch.Tensor, target: torch.Tensor, initial: torch.Tensor,
                           lower: torch.Tensor, upper: torch.Tensor,
                           dt: float, time_constant: float, rate: float) -> torch.Tensor:
    # 一阶指数趋近局部目标，同时限制单步变化率与刚度上下界
    delta = (target-current) * (1-math.exp(-dt/time_constant))
    max_delta = initial * rate * dt
    return torch.maximum(lower, torch.minimum(upper, current + delta.clamp(-max_delta, max_delta)))


def play_hysteresis_step(state: torch.Tensor, signal: torch.Tensor,
                         half_width: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(state, signal-half_width), signal+half_width)


def completion_cost_terms(
    cfg: InsertionCfg, xy: torch.Tensor, z_disp: torch.Tensor,
    target_depth: torch.Tensor, radius: float, tilt: torch.Tensor, yaw: torch.Tensor,
    wrench: torch.Tensor, actions: torch.Tensor, previous_actions: torch.Tensor,
    success: torch.Tensor, safety_failure: torch.Tensor, timeout: torch.Tensor,
    *, tip_center_z: torch.Tensor | None = None,
    hole_force: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    dt = cfg.physics_dt*cfg.decimation
    remaining = z_disp.clamp_min(0)/target_depth
    # XY 单独平方以放宽中心附近探索，剩余深度保持线性
    distance = (xy/(radius*cfg.xy_cost_radius_fraction)).square() + remaining
    distance = cfg.position_cost_cap*torch.tanh(distance/cfg.position_cost_cap)
    reference = math.radians(cfg.orientation_cost_reference_deg)
    tilt_error, yaw_error = (tilt/reference).square(), (yaw/reference).square()
    # 倾角权重在最低点进入前 4 mm 内线性增强，与最终目标深度无关
    tilt_progress = ((target_depth-z_disp)/0.004).clamp(0, 1)
    tilt_weight = cfg.preinsertion_tilt_weight + (1-cfg.preinsertion_tilt_weight)*tilt_progress
    tilt_cost, yaw_cost = tilt_error/(1+tilt_error), yaw_error/(1+yaw_error)
    orientation = tilt_weight*tilt_cost + yaw_cost
    force = ((torch.linalg.vector_norm(wrench[:, :3], dim=-1)-cfg.force_cost_free_n)
             / (cfg.max_force-cfg.force_cost_free_n)).clamp(0, 1)
    moment = ((torch.linalg.vector_norm(wrench[:, 3:], dim=-1)-cfg.moment_cost_free_nm)
              / (cfg.max_moment-cfg.moment_cost_free_nm)).clamp(0, 1)
    if cfg.inserted_load_cost and tip_center_z is not None and hole_force is not None:
        inserted = tip_center_z < -cfg.inserted_load_depth_m
        lateral_force = (torch.linalg.vector_norm(hole_force[:, :2], dim=-1)
                         / cfg.max_force).clamp(0, 1)
        full_moment = (torch.linalg.vector_norm(wrench[:, 3:], dim=-1)
                       / cfg.max_moment).clamp(0, 1)
        force = torch.where(inserted, lateral_force, force)
        moment = torch.where(inserted, full_moment, moment)
    applied = actions.clamp(-1, 1)
    previous = previous_actions.clamp(-1, 1)
    return {
        "time_penalty": torch.full_like(xy, -dt*cfg.time_cost_per_second),
        "position_penalty": -dt*cfg.position_cost_per_second*distance,
        "orientation_penalty": -dt*cfg.orientation_cost_per_second*orientation,
        "force_penalty": -dt*cfg.force_cost_per_second*force.square(),
        "moment_penalty": -dt*cfg.moment_cost_per_second*moment.square(),
        "action_penalty": -dt*cfg.action_cost_per_second*torch.linalg.vector_norm(applied, dim=-1),
        "action_change_penalty": -dt*cfg.action_change_cost_per_second*torch.linalg.vector_norm(
            applied-previous, dim=-1),
        "completion_bonus": cfg.completion_bonus*success.float(),
        "safety_failure_penalty": -cfg.safety_failure_penalty*safety_failure.float(),
        "timeout_penalty": -cfg.timeout_penalty*(timeout & ~success & ~safety_failure).float(),
    }


class PlantGroup:
    def __init__(self, shape: str, cfg: InsertionCfg, device: str):
        self.cfg, self.device, self.n = cfg, device, cfg.envs_per_shape
        self.last_reward_terms: dict[str, torch.Tensor] = {}
        self.clearance = cfg.active_clearance_levels[0]
        self.shape = shape
        self._build_simulation()
        self.generator = torch.Generator(device=device).manual_seed(
            cfg.seed + ALL_SHAPES.index(shape) * 10007)
        self.difficulty = 0.25 if cfg.training else 1.0
        self.noise_scale = cfg.noise_scale(0)
        self.episode_noise_scale = torch.full((self.n,), self.noise_scale, device=device)
        self.episode_clearance = torch.full((self.n,), self.clearance, device=device)
        self.nominal_offset = torch.tensor([0, 0, -0.05 * cfg.scale],
                                           device=device).expand(self.n, 3)
        self.offset = self.nominal_offset.clone()
        self.sensor_offset = torch.tensor([0, 0, 0.03 * cfg.scale],
                                          device=device).expand(self.n, 3)
        self.identity = torch.tensor([1., 0, 0, 0], device=device).expand(self.n, 4)
        self.goal = torch.zeros(self.n, 3, device=device)
        self.length = torch.zeros(self.n, device=device, dtype=torch.long)
        self.position_delay_frames = torch.zeros_like(self.length)
        self.position_history = torch.zeros(self.n, 2, 3, device=device)
        self.position_history_length = torch.full_like(self.length, -1)
        self.hold = torch.zeros_like(self.length)
        self.peak_force = torch.zeros(self.n, device=device)
        self.peak_moment = torch.zeros_like(self.peak_force)
        self.last_action = torch.zeros(self.n, 6, device=device)
        self.wrench = torch.zeros(self.n, 6, device=device)
        self.wrench_bias = torch.zeros_like(self.wrench)
        self.raw_wrench = torch.zeros_like(self.wrench)
        self.filtered_sensor = torch.zeros_like(self.wrench)
        self.hysteresis_sensor = torch.zeros_like(self.wrench)
        self.hysteresis_half_width = torch.tensor(
            [cfg.max_force*cfg.wrench_hysteresis_fraction/2]*3
            + [cfg.max_moment*cfg.wrench_hysteresis_fraction/2]*3,
            device=device)
        self.k = torch.zeros(self.n, 6, device=device)
        self.d = torch.zeros_like(self.k)
        self.initial_k = torch.zeros_like(self.k, dtype=torch.float64)
        self.walk_lower = torch.zeros_like(self.initial_k)
        self.walk_upper = torch.zeros_like(self.initial_k)
        self.walk_target = torch.zeros_like(self.initial_k)
        self.walk_current = torch.zeros_like(self.initial_k)
        self.damping_ratio = torch.ones_like(self.k)
        self.walk_generator = torch.Generator(device=device).manual_seed(
            cfg.seed + ALL_SHAPES.index(shape) * 10007 + 500003)
        self.gain_q = self.identity.clone()
        self.mass = torch.ones(self.n, 1, device=device)
        self.inertia = torch.zeros(self.n, 3, device=device)
        self.down_mass = torch.zeros_like(self.mass)
        self.down_inertia = torch.zeros_like(self.inertia)
        self.calibration = torch.zeros(self.n, 3, device=device)
        self.sensor_error = self.identity.clone()
        self.bias = torch.zeros(self.n, 3, device=device)
        self.delay = torch.zeros(self.n, device=device, dtype=torch.long)
        self.targets_p = torch.zeros(3, self.n, 3, device=device)
        self.targets_q = self.identity.unsqueeze(0).repeat(3, 1, 1)
        self.physics_p_goal = torch.zeros(self.n, 3, device=device)
        self.physics_q_goal = self.identity.clone()
        # 使用张量视图避免 TorchArray 索引写入触发跨流切换，以便被 CUDA Graph 捕获
        self.applied_wrench = self.sim.data.xfrc_applied[:, self.body_id]
        self.physics_graph: torch.cuda.CUDAGraph | None = None
        self.sensor_graph: torch.cuda.CUDAGraph | None = None
        self.physics_rollout_graph: torch.cuda.CUDAGraph | None = None
        self.reset_constants_graph: torch.cuda.CUDAGraph | None = None
        self.action_graph: torch.cuda.CUDAGraph | None = None
        self.outcome_graph: torch.cuda.CUDAGraph | None = None
        self.step_actions = torch.zeros(self.n, 6, device=device)
        self.outcome_outputs: tuple[torch.Tensor, torch.Tensor, dict] | None = None
        self.outcome_reward_terms: dict[str, torch.Tensor] = {}
        self.observation_graph: torch.cuda.CUDAGraph | None = None
        self.observation_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.observation_noise = (
            torch.zeros(self.n, 6, device=device),
            torch.zeros(self.n, 3, device=device),
            torch.zeros(self.n, 3, device=device),
        )
        self.physics_graph_signature: tuple[int, float, str, float, float, float, float] | None = None
        self.reset(torch.arange(self.n, device=device))

    def _build_simulation(self) -> None:
        cfg = self.cfg
        self.geometry = geometry(self.shape, self.clearance, cfg.scale, cfg.circle_sides,
                                 cfg.plate_outer_radius_m)
        self.model = build_model(self.geometry)
        self.body_id = self.model.body("tool").id
        self.sim = Simulation(
            self.n, SimulationCfg(nconmax=512 if self.shape == "circle" else 64,
                                 njmax=2048 if self.shape == "circle" else 256,
                                 mujoco=MujocoCfg(timestep=cfg.physics_dt, gravity=(0, 0, 0),
                                                 iterations=50, ls_iterations=20)),
            model=self.model, device=self.device,
        )
        self.sim.expand_model_fields(("body_mass", "body_inertia", "geom_friction", "geom_pos"))
        self.peg_geom_ids = (self.model.geom_bodyid == self.body_id).nonzero()[0].tolist()
        self.default_geom_pos = torch.tensor(
            self.model.geom_pos, device=self.device, dtype=torch.float32)
        self.peg_polygon = torch.tensor(
            self.geometry.peg_polygon, dtype=torch.float32, device=self.device)
        self.hole_polygon = torch.tensor(
            self.geometry.hole_polygon, dtype=torch.float32, device=self.device)

    def uniform(self, n: int, columns: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, columns, generator=self.generator,
                                             device=self.device)

    def normal(self, n: int, columns: int, std: float) -> torch.Tensor:
        if std == 0:
            return torch.zeros(n, columns, device=self.device)
        return torch.randn(n, columns, generator=self.generator, device=self.device) * std

    def sample_wrench_bias(self, n: int, amplitude: float) -> torch.Tensor:
        if self.cfg.wrench_bias_distribution == "per_axis":
            return self.uniform(n, 3, -amplitude, amplitude)
        azimuth = self.uniform(n, 1, -math.pi, math.pi)
        z = self.uniform(n, 1, -1., 1.)
        xy = (1-z.square()).clamp_min(0).sqrt()
        direction = torch.cat((xy*azimuth.cos(), xy*azimuth.sin(), z), dim=-1)
        return direction*self.uniform(n, 1, 0., amplitude)

    def set_clearance(self, clearance: float) -> None:
        if math.isclose(clearance, self.clearance, rel_tol=0., abs_tol=1e-12):
            return
        qpos = self.sim.data.qpos[:].clone()
        qvel = self.sim.data.qvel[:].clone()
        friction = self.sim.model.geom_friction[:, :, 0].clone()
        self.clearance = clearance
        self._build_simulation()
        self.sim.model.body_mass[:, self.body_id] = self.mass[:, 0]
        self.sim.model.body_inertia[:, self.body_id] = self.inertia
        self.sim.model.geom_friction[:, :, 0] = friction
        grasp = self.offset - self.nominal_offset
        for geom_id in self.peg_geom_ids:
            self.sim.model.geom_pos[:, geom_id] = self.default_geom_pos[geom_id] + grasp
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.data.xfrc_applied[:] = 0
        self.sim.forward()
        self.applied_wrench = self.sim.data.xfrc_applied[:, self.body_id]
        self.physics_graph = None
        self.sensor_graph = None
        self.physics_rollout_graph = None
        self.reset_constants_graph = None
        self.action_graph = None
        self.outcome_graph = None
        self.outcome_outputs = None
        self.outcome_reward_terms = {}
        self.physics_graph_signature = None
        self.observation_graph = None
        self.observation_outputs = None
        pos, q, _, _ = self.state()
        in_hole = (tip_pose(pos, q, self.offset)[:, 2] <= 0.002).nonzero().flatten()
        if len(in_hole):
            self.reset(in_hole)

    def reset(self, ids: torch.Tensor) -> None:
        n, cfg = len(ids), self.cfg
        self.episode_noise_scale[ids] = self.noise_scale
        self.episode_clearance[ids] = self.clearance
        self.sim.reset(ids)
        low_depth, high_depth = cfg.target_depth_bounds_m
        if low_depth == high_depth:
            self.goal[ids, 2] = -low_depth
        else:
            self.goal[ids, 2] = -self.uniform(n, 1, low_depth, high_depth)[:, 0]
        if cfg.dynamics == "train":
            low, high, rot_low, rot_high = 300., 2500., 4., 12.
        elif cfg.dynamics == "soft":
            low, high, rot_low, rot_high = 300., 600., 1.5, 3.
        elif cfg.dynamics == "stiff":
            low, high, rot_low, rot_high = 3000., 5000., 15., 22.
        else:
            low = high = 1500.
            rot_low = rot_high = 8.
        self.k[ids] = torch.cat((self.uniform(n, 3, low, high),
                                 self.uniform(n, 3, rot_low, rot_high)), -1)
        lower = torch.tensor([low]*3 + [rot_low]*3, device=self.device)
        upper = torch.tensor([high]*3 + [rot_high]*3, device=self.device)
        if cfg.training and cfg.dynamics == "train":
            center = torch.tensor([1500.]*3 + [8.]*3, device=self.device)
            self.k[ids] = center + self.difficulty * (self.k[ids] - center)
            lower = center + self.difficulty * (lower-center)
            upper = center + self.difficulty * (upper-center)
        self.initial_k[ids] = self.k[ids].double()
        self.walk_lower[ids] = torch.maximum(lower, self.initial_k[ids]*(1-cfg.impedance_walk_fraction))
        self.walk_upper[ids] = torch.minimum(upper, self.initial_k[ids]*(1+cfg.impedance_walk_fraction))
        self.walk_target[ids] = self.initial_k[ids]
        self.walk_current[ids] = self.initial_k[ids]
        self.mass[ids] = self.uniform(n, 1, 0.7, 1.5)
        inertia_parts = self.uniform(n, 3, 0.001, 0.0025)
        self.inertia[ids] = (inertia_parts.sum(-1, keepdim=True)-inertia_parts) * cfg.scale**2
        if cfg.dynamics == "nominal":
            self.mass[ids] = 1
            self.inertia[ids] = 0.003 * cfg.scale**2
        inertia6 = torch.cat((self.mass[ids].expand(n, 3), self.inertia[ids]), -1)
        self.damping_ratio[ids] = self.uniform(n, 6, 0.8, 1.3)
        self.d[ids] = 2 * torch.sqrt(self.k[ids] * inertia6) * self.damping_ratio[ids]
        self.gain_q[ids] = rotvec_quat(self.uniform(n, 3, -0.5, 0.5))
        self.down_mass[ids] = self.uniform(n, 1, 0.2, 0.4)
        down_parts = self.uniform(n, 3, 0.00015, 0.00035)
        self.down_inertia[ids] = (down_parts.sum(-1, keepdim=True)-down_parts) * cfg.scale**2
        if cfg.dynamics == "nominal":
            self.damping_ratio[ids] = 1
            self.d[ids] = 2 * torch.sqrt(self.k[ids] * inertia6)
            self.gain_q[ids] = self.identity[ids]
            self.down_mass[ids] = 0.3
            self.down_inertia[ids] = 0.0005 * cfg.scale**2
        self.sim.model.body_mass[ids, self.body_id] = self.mass[ids, 0]
        self.sim.model.body_inertia[ids, self.body_id] = self.inertia[ids]
        friction = self.uniform(n, 1, 0.25, 0.6)
        if cfg.dynamics == "nominal":
            friction.fill_(0.4)
        self.sim.model.geom_friction[ids, :, 0] = friction
        if cfg.grasp_error_distribution == "gaussian":
            grasp = self.normal(n, 3, cfg.grasp_position_error)
        else:
            grasp = self.uniform(n, 3, -cfg.grasp_position_error, cfg.grasp_position_error)
        grasp = grasp*self.noise_scale
        self.offset[ids] = self.nominal_offset[ids] + grasp
        for geom_id in self.peg_geom_ids:
            self.sim.model.geom_pos[ids, geom_id] = self.default_geom_pos[geom_id] + grasp
        if self.reset_constants_graph is not None and cfg.physics_cuda_graph:
            self.reset_constants_graph.replay()
        else:
            self.sim.recompute_constants(RecomputeLevel.set_const)
            if cfg.physics_cuda_graph:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    graph = torch.cuda.CUDAGraph()
                    with wp.ScopedStream(wp.stream_from_torch(stream)):
                        with torch.cuda.graph(graph, stream=stream):
                            with wp.ScopedCapture(external=True, capture_mode=wp.CaptureMode.GLOBAL):
                                self.sim.recompute_constants(RecomputeLevel.set_const)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                self.reset_constants_graph = graph
        self.calibration[ids] = self.uniform(n, 3, -cfg.tip_calibration_error,
                                            cfg.tip_calibration_error)*self.noise_scale
        self.bias[ids] = self.uniform(n, 3, -cfg.position_bias, cfg.position_bias)*self.noise_scale
        self.sensor_error[ids] = rotvec_quat(self.uniform(
            n, 3, -cfg.sensor_rotation_error, cfg.sensor_rotation_error)*self.noise_scale)
        if cfg.force_bias or cfg.moment_bias:
            self.wrench_bias[ids] = torch.cat((
                self.sample_wrench_bias(n, cfg.force_bias),
                self.sample_wrench_bias(n, cfg.moment_bias)), -1) * self.noise_scale
        else:
            self.wrench_bias[ids] = 0
        self.delay[ids] = torch.randint(0, 3, (n,), generator=self.generator, device=self.device)
        if cfg.dynamics == "nominal":
            self.delay[ids] = 0
        pos = self.uniform(n, 3, -cfg.initial_xy, cfg.initial_xy) * self.difficulty
        pos[:, 2] = self.uniform(n, 1, cfg.initial_z_min, cfg.initial_z_max)[:, 0]
        q = self.sample_initial_orientation(n)
        if cfg.initial_plane_margin:
            for attempt in range(16):
                minimum_z = self.initial_minimum_tip_height(q)
                invalid = minimum_z > cfg.initial_z_max
                if not bool(invalid.any()):
                    break
                if attempt == 15:
                    raise RuntimeError("Cannot sample collision-free initial pose within Z bounds after 16 attempts")
                q[invalid] = self.sample_initial_orientation(int(invalid.sum()))
            height_fraction = torch.zeros_like(pos[:, 2])
            if cfg.initial_z_max > cfg.initial_z_min:
                height_fraction = (pos[:, 2]-cfg.initial_z_min) / (
                    cfg.initial_z_max-cfg.initial_z_min)
            pos[:, 2] = minimum_z + height_fraction * (cfg.initial_z_max-minimum_z)
        self.sim.data.qpos[ids, :3] = pos - rotate(q, self.offset[ids])
        self.sim.data.qpos[ids, 3:7] = q
        self.sim.data.qvel[ids] = 0
        self.sim.data.xfrc_applied[ids] = 0
        self.sim.forward()
        nominal_tip = tip_pose(self.sim.data.qpos[ids, :3], q,
                               self.nominal_offset[ids] + self.calibration[ids])
        self.targets_p[:, ids] = nominal_tip.unsqueeze(0)
        self.targets_q[:, ids] = q.unsqueeze(0)
        for buffer in (self.length, self.hold, self.peak_force, self.peak_moment,
                       self.last_action, self.wrench, self.raw_wrench, self.filtered_sensor,
                       self.hysteresis_sensor):
            buffer[ids] = 0
        if cfg.position_observation_delay:
            self.reset_position_observation(ids, nominal_tip)

    def reset_position_observation(self, ids: torch.Tensor, observed_tip: torch.Tensor) -> None:
        n = len(ids)
        self.position_delay_frames[ids] = torch.randint(
            0, 2, (n,), generator=self.generator, device=self.device)
        position_noise = self.normal(n, 3, self.cfg.position_noise)*self.noise_scale
        error = self.goal[ids] + self.bias[ids] - observed_tip - position_noise
        self.position_history[ids] = error[:, None, :].expand(-1, 2, -1)
        self.position_history_length[ids] = self.length[ids]

    def sample_initial_orientation(self, n: int) -> torch.Tensor:
        cfg = self.cfg
        rv = self.uniform(n, 3, -cfg.initial_tilt, cfg.initial_tilt) * self.difficulty
        rv[:, 2] = self.uniform(n, 1, -cfg.initial_yaw, cfg.initial_yaw)[:, 0] * self.difficulty
        return rotvec_quat(rv)

    def minimum_peg_height_offset(self, q: torch.Tensor) -> torch.Tensor:
        # 单位四元数旋转矩阵的第三行，即孔 Z 轴在插头坐标系中的方向
        w, x, y, z = q.unbind(dim=-1)
        local_vertical = torch.stack((2*(x*z-w*y), 2*(y*z+w*x),
                                      1-2*(x.square()+y.square())), dim=-1)
        if self.shape == "circle":
            radial_minimum = -(self.geometry.radius-self.clearance/2) * torch.linalg.vector_norm(
                local_vertical[:, :2], dim=-1)
        else:
            radial_minimum = (local_vertical[:, :2] @ self.peg_polygon.T).amin(dim=-1)
        return radial_minimum + (local_vertical[:, 2]*self.geometry.peg_length).clamp_max(0)

    def initial_minimum_tip_height(self, q: torch.Tensor) -> torch.Tensor:
        lowest = self.minimum_peg_height_offset(q)
        return (self.cfg.initial_plane_margin-lowest).clamp_min(self.cfg.initial_z_min)

    def state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = self.sim.data.qpos[:, :3]
        q = canonical_quat(self.sim.data.qpos[:, 3:7])
        return pos, q, self.sim.data.qvel[:, :3], rotate(q, self.sim.data.qvel[:, 3:6])

    def impedance(self, error: torch.Tensor, velocity: torch.Tensor,
                  part: slice) -> torch.Tensor:
        frame_inv = quat_inv(self.gain_q)
        local_e, local_v = rotate(frame_inv, error), rotate(frame_inv, velocity)
        return rotate(self.gain_q, self.k[:, part] * local_e - self.d[:, part] * local_v)

    def sample_impedance_walk_target(self) -> None:
        if self.cfg.dynamics == "nominal" or self.cfg.impedance_walk_fraction == 0:
            return
        interval = round(self.cfg.impedance_walk_interval_seconds / 0.02)
        ids = (self.length % interval == 0).nonzero().flatten()
        if len(ids):
            u = torch.rand(len(ids), 6, generator=self.walk_generator, device=self.device)
            self.walk_target[ids] = self.walk_lower[ids] + u * (
                self.walk_upper[ids]-self.walk_lower[ids])

    def update_impedance_walk(self) -> None:
        cfg = self.cfg
        if cfg.dynamics == "nominal" or cfg.impedance_walk_fraction == 0:
            return
        # 使用 float64 累积物理子步，避免浮点舍入系统性突破变化率上限
        self.walk_current.copy_(bounded_stiffness_step(
            self.walk_current, self.walk_target, self.initial_k, self.walk_lower, self.walk_upper,
            cfg.physics_dt, cfg.impedance_walk_time_constant_seconds,
            cfg.impedance_walk_rate_per_second))
        self.k.copy_(self.walk_current)
        inertia6 = torch.cat((self.mass.expand(self.n, 3), self.inertia), -1)
        self.d.copy_(2 * self.damping_ratio * torch.sqrt(self.k * inertia6))

    def update_sensor(self) -> None:
        _, q, _, _ = self.state()
        acc = self.sim.data.qacc[:, :3]
        alpha_local = self.sim.data.qacc[:, 3:6]
        omega_local = self.sim.data.qvel[:, 3:6]
        contact_f = self.sim.data.qfrc_constraint[:, :3]
        contact_m = rotate(q, self.sim.data.qfrc_constraint[:, 3:6])
        inertial_m = rotate(q, self.down_inertia * alpha_local
                            + torch.linalg.cross(omega_local, self.down_inertia * omega_local))
        f = contact_f - self.down_mass * acc
        m_com = contact_m - inertial_m
        # 传感器下游质心位于刚体原点，等效机械臂惯量不计入腕部传感器负载
        wrist = transform_wrench(f, m_com, quat_inv(q), self.sensor_offset)
        alpha = 1 - math.exp(-self.cfg.physics_dt / self.cfg.filter_seconds)
        self.filtered_sensor += alpha * (wrist - self.filtered_sensor)
        self.hysteresis_sensor.copy_(play_hysteresis_step(
            self.hysteresis_sensor, self.filtered_sensor, self.hysteresis_half_width))
        r_st = self.nominal_offset + self.calibration - self.sensor_offset
        self.wrench.copy_(transform_wrench(self.hysteresis_sensor[:, :3],
                                          self.hysteresis_sensor[:, 3:],
                                          self.sensor_error, rotate(self.sensor_error, r_st)))
        self.raw_wrench.copy_(transform_wrench(wrist[:, :3], wrist[:, 3:],
                                              self.identity, self.offset - self.sensor_offset))

    def observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        noise_scale = torch.tensor([self.cfg.force_noise]*3 + [self.cfg.moment_noise]*3,
                                   device=self.device)
        noise = torch.randn(self.n, 6, generator=self.generator,
                            device=self.device) * noise_scale*self.noise_scale
        position_noise = self.normal(self.n, 3, self.cfg.position_noise)*self.noise_scale
        rotation_noise = self.normal(self.n, 3, self.cfg.rotation_noise)*self.noise_scale
        self.record_position_observation(position_noise)
        if not self.cfg.observation_cuda_graph:
            return self.observations_from_noise(noise, position_noise, rotation_noise)
        for target, value in zip(self.observation_noise,
                                 (noise, position_noise, rotation_noise), strict=True):
            target.copy_(value)
        if self.observation_graph is None:
            self.observations_from_noise(*self.observation_noise)
            with capture_graph(self.device) as graph:
                outputs = self.observations_from_noise(*self.observation_noise)
            self.observation_graph, self.observation_outputs = graph, outputs
        self.observation_graph.replay()
        assert self.observation_outputs is not None
        # 克隆输出缓冲区，防止外部持有引用时被后续图重放覆盖
        actor, critic = self.observation_outputs
        return actor.clone(), critic.clone()

    def record_position_observation(self, position_noise: torch.Tensor) -> None:
        if not self.cfg.position_observation_delay:
            return
        pos, q, _, _ = self.state()
        observed_tip = tip_pose(pos, q, self.nominal_offset + self.calibration)
        error = self.goal + self.bias - observed_tip - position_noise
        advanced = self.length != self.position_history_length
        history = torch.cat((error[:, None, :], self.position_history[:, :1]), dim=1)
        self.position_history.copy_(torch.where(
            advanced[:, None, None], history, self.position_history))
        self.position_history_length.copy_(self.length)

    def observations_from_noise(self, noise: torch.Tensor, position_noise: torch.Tensor,
                                rotation_noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos, q, v, omega = self.state()
        true_tip = tip_pose(pos, q, self.offset)
        observed_tip = tip_pose(pos, q, self.nominal_offset + self.calibration)
        noisy_tip = observed_tip + position_noise
        noisy_q = quat_mul(rotvec_quat(rotation_noise), q)
        obs = actor_observation(noisy_tip, noisy_q, self.goal + self.bias,
                                self.identity, self.wrench + self.wrench_bias + noise)
        if self.cfg.position_observation_delay:
            delayed = self.position_history.gather(
                1, self.position_delay_frames[:, None, None].expand(-1, 1, 3)).squeeze(1)
            obs = torch.cat((delayed, obs[:, 3:]), dim=-1)
        clean = actor_observation(true_tip, q, self.goal, self.identity, self.raw_wrench)
        shape_id = torch.nn.functional.one_hot(
            torch.full((self.n,), ALL_SHAPES.index(self.shape), device=self.device),
            len(ALL_SHAPES)).float()
        critic = torch.cat((clean, v, omega, self.k, self.d, self.mass, self.inertia,
                            self.down_mass, self.down_inertia, self.calibration,
                            quat_rotvec(self.sensor_error), self.bias,
                            self.offset-self.nominal_offset, shape_id), -1)
        return obs, critic

    def alignment(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(self.n, 3, device=self.device)
        z[:, 2] = 1
        axis = rotate(q, z)
        tilt = torch.atan2(torch.linalg.vector_norm(axis[:, :2], dim=-1), axis[:, 2])
        count = symmetry_count(self.shape)
        if count == 0:
            yaw = torch.zeros_like(tilt)
        else:
            yaw = torch.atan2(2*(q[:, 0]*q[:, 3] + q[:, 1]*q[:, 2]),
                              1 - 2*(q[:, 2]**2 + q[:, 3]**2))
            period = 2 * math.pi / count
            yaw = torch.remainder(yaw + period/2, period) - period/2
        return tilt, yaw.abs()

    def physics_control(self) -> None:
        self.update_impedance_walk()
        pos, q, v, omega = self.state()
        r_ct = rotate(q, self.nominal_offset + self.calibration)
        p_tip = pos + r_ct
        v_tip = v + torch.linalg.cross(omega, r_ct)
        force = self.impedance(self.physics_p_goal - p_tip, v_tip, slice(0, 3)).clamp(-40, 40)
        angular_error = quat_rotvec(quat_mul(self.physics_q_goal, quat_inv(q)))
        moment_tip = self.impedance(angular_error, omega, slice(3, 6)).clamp(-3, 3)
        self.applied_wrench[:, :3].copy_(force)
        self.applied_wrench[:, 3:].copy_(moment_tip + torch.linalg.cross(r_ct, force))

    def physics_sensor(self) -> None:
        self.update_sensor()
        self.peak_force.copy_(torch.maximum(self.peak_force,
            torch.linalg.vector_norm(self.raw_wrench[:, :3], dim=-1)))
        self.peak_moment.copy_(torch.maximum(self.peak_moment,
            torch.linalg.vector_norm(self.raw_wrench[:, 3:], dim=-1)))

    def advance_physics(self) -> None:
        cfg = self.cfg
        signature = (cfg.decimation, cfg.physics_dt, cfg.dynamics,
                     cfg.impedance_walk_fraction, cfg.impedance_walk_time_constant_seconds,
                     cfg.impedance_walk_rate_per_second, cfg.filter_seconds)
        use_graph = (cfg.physics_cuda_graph and self.physics_graph is not None
                     and self.physics_graph_signature == signature)
        if use_graph and self.physics_rollout_graph is not None and not self.sim.nan_guard.enabled:
            self.physics_rollout_graph.replay()
            return
        self.physics_rollout_graph = None
        for _ in range(cfg.decimation):
            if use_graph:
                assert self.physics_graph is not None
                self.physics_graph.replay()
            else:
                self.physics_control()
            self.sim.step()
            if use_graph:
                assert self.sensor_graph is not None
                self.sensor_graph.replay()
            else:
                self.physics_sensor()
        if cfg.physics_cuda_graph and not use_graph:
            # 先完成首步真实预热再捕获计算图，图捕获本身不推进仿真步
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                control_graph, sensor_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(control_graph, stream=stream):
                    self.physics_control()
                with torch.cuda.graph(sensor_graph, stream=stream):
                    self.physics_sensor()
                torch.cuda.current_stream(self.device).wait_stream(stream)
            self.physics_graph, self.sensor_graph = control_graph, sensor_graph
            self.physics_graph_signature = signature
            if self.sim.use_cuda_graph and self.sim.step_graph is not None and not self.sim.nan_guard.enabled:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    rollout_graph = torch.cuda.CUDAGraph()
                    # 在同一流中捕获原始操作，保持每个 1 ms 子步的读写顺序
                    with wp.ScopedStream(wp.stream_from_torch(stream)):
                        with torch.cuda.graph(rollout_graph, stream=stream):
                            with wp.ScopedCapture(external=True, capture_mode=wp.CaptureMode.GLOBAL):
                                for _ in range(cfg.decimation):
                                    self.physics_control()
                                    mjwarp.step(self.sim.wp_model, self.sim.wp_data)
                                    self.physics_sensor()
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                self.physics_rollout_graph = rollout_graph

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite policy action; no silent reset")
        if self.cfg.physics_cuda_graph:
            self.step_actions.copy_(actions)
            if self.action_graph is None:
                self.prepare_actions(self.step_actions)
                with capture_graph(self.device) as graph:
                    self.prepare_actions(self.step_actions)
                self.action_graph = graph
            else:
                self.action_graph.replay()
        else:
            self.prepare_actions(actions)
        self.sample_impedance_walk_target()
        self.advance_physics()
        if not torch.isfinite(self.sim.data.qpos[:]).all() or not torch.isfinite(self.raw_wrench).all():
            raise FloatingPointError(f"invalid physical state in {self.shape}")
        if not self.cfg.physics_cuda_graph:
            return self.step_outcome(actions)
        if self.outcome_graph is None:
            result = self.step_outcome(self.step_actions)
            terms = self.last_reward_terms
            with capture_graph(self.device) as graph:
                self.outcome_outputs = self.step_outcome(self.step_actions)
            self.outcome_reward_terms = self.last_reward_terms
            self.last_reward_terms = terms
            self.outcome_graph = graph
            return result
        self.outcome_graph.replay()
        self.last_reward_terms = self.outcome_reward_terms
        assert self.outcome_outputs is not None
        reward, done, info = self.outcome_outputs
        # 图输出下次重放会被覆盖，不能泄露给 rollout 或终局信息的持有者
        return reward.clone(), done.clone(), {key: value.clone() for key, value in info.items()}

    def prepare_actions(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        pos, q, _, _ = self.state()
        p_target, q_target = action_target(tip_pose(pos, q, self.nominal_offset + self.calibration),
                                           q, actions, cfg.translation_limit, cfg.rotation_limit)
        self.targets_p.copy_(torch.cat((p_target.unsqueeze(0), self.targets_p[:2]), 0))
        self.targets_q.copy_(torch.cat((q_target.unsqueeze(0), self.targets_q[:2]), 0))
        ids = torch.arange(self.n, device=self.device)
        self.physics_p_goal.copy_(self.targets_p[self.delay, ids])
        self.physics_q_goal.copy_(self.targets_q[self.delay, ids])

    def step_outcome(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cfg = self.cfg
        pos, q, _, _ = self.state()
        tip = tip_pose(pos, q, self.offset)
        xy = torch.linalg.vector_norm(tip[:, :2], dim=-1)
        tilt, yaw = self.alignment(q)
        lowest_z = tip[:, 2] + self.minimum_peg_height_offset(q)
        z_disp = lowest_z - self.goal[:, 2]
        success = tip[:, 2] <= self.goal[:, 2] + 0.0002
        if cfg.success_opening_required:
            inside = ((tip[:, :2].square().sum(-1) <= self.geometry.radius**2)
                      if self.shape == "circle" else points_in_polygon(tip[:, :2], self.hole_polygon))
            success = success & inside
        if cfg.success_alignment_required:
            success = (success & (xy < self.clearance * 0.4)
                       & (tilt < self.clearance / self.geometry.peg_length * 0.4)
                       & (yaw < self.clearance / self.geometry.radius * 0.4))
        self.hold.copy_(torch.where(success, self.hold + 1, torch.zeros_like(self.hold)))
        success = self.hold >= cfg.success_hold_steps
        excessive = cfg.terminate_on_overload & (
            (self.peak_force > cfg.max_force) | (self.peak_moment > cfg.max_moment))
        escaped = ((xy > cfg.workspace_xy_radius_m) | (tip[:, 2] > 0.03 * cfg.scale)
                   | (tip[:, 2] < -self.geometry.depth) | (tilt > math.radians(35)))
        self.length += 1
        timeout = self.length >= math.ceil(cfg.episode_seconds / 0.02)
        done = success | excessive | escaped | timeout
        success = success & ~excessive & ~escaped
        terms = completion_cost_terms(
            cfg, xy, z_disp, -self.goal[:, 2], self.geometry.radius, tilt, yaw,
            self.wrench, actions, self.last_action, success, excessive | escaped, timeout,
            tip_center_z=tip[:, 2], hole_force=rotate(q, self.wrench[:, :3]))
        self.last_reward_terms = terms
        reward = torch.stack(list(terms.values())).sum(0)
        self.last_action.copy_(actions)
        # 安全终止优先于成功判定，仅在未触发成功或安全越界时报告超时
        reason = torch.where(success, 1, torch.where(excessive, 2, torch.where(escaped, 3,
                             torch.where(timeout, 4, 0))))
        info = {"reason": reason, "length": self.length.clone(),
                "tip_z_m": tip[:, 2].clone(), "tilt_rad": tilt.clone(),
                "target_depth_m": -self.goal[:, 2].clone(),
                "peak_force": self.peak_force.clone(), "peak_moment": self.peak_moment.clone(),
                "time_outs": timeout & ~success & ~excessive & ~escaped}
        return reward, done, info


class InsertionEnv(VecEnv):
    def __init__(self, cfg: InsertionCfg, device: str = "cuda:0",
                 simulation_devices: tuple[str, ...] | None = None):
        self.cfg: dict[str, Any] = asdict(cfg)
        self.device: str = device
        devices = (device,) if simulation_devices is None else simulation_devices
        if not devices:
            raise ValueError("simulation_devices must not be empty")
        if any(n % len(devices) for n in cfg.environment_counts):
            raise ValueError("each shape count must be divisible by the simulation device count")
        self.groups = [
            PlantGroup(s, replace(cfg, shapes=(s,), shape_env_counts=(),
                                  envs_per_shape=count // len(devices),
                                  seed=cfg.seed + i*1000003), simulation_device)
            for s, count in zip(cfg.shapes, cfg.environment_counts, strict=True)
            for i, simulation_device in enumerate(devices)]
        self.num_envs = cfg.envs_per_shape * len(cfg.shapes)
        self.num_actions = 6
        self.max_episode_length = math.ceil(cfg.episode_seconds / 0.02)
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.policy_steps = 0
        self.total_transitions = 0
        use_success_curriculum = cfg.training and (
            cfg.noise_curriculum_mode == "success" or cfg.clearance_curriculum)
        noise_levels = (10 if cfg.noise_curriculum and cfg.noise_curriculum_mode == "success"
                        else 0)
        self.noise_curriculum = (
            SuccessNoiseCurriculum(cfg.shapes, cfg.active_clearance_levels, noise_levels)
            if use_success_curriculum else None)
        self.curriculum_pending: list[torch.Tensor] = []
        self.curriculum_cohort: torch.Tensor | None = None
        self.curriculum_cohort_seen: torch.Tensor | None = None
        self.curriculum_distributed = False
        self.curriculum_rollout_steps = 256
        self.curriculum_shape_ids = torch.cat([
            torch.full((g.n,), cfg.shapes.index(g.shape), device=device) for g in self.groups])

    def record_curriculum_outcomes(self, outcomes: torch.Tensor) -> None:
        if self.noise_curriculum is not None:
            self.curriculum_pending.append(outcomes)

    def advance_noise_curriculum(self) -> dict:
        if self.noise_curriculum is None or not self.curriculum_pending:
            return {}
        # 每个 rollout 边界统一汇总一次回合结果，不在单步内频繁同步
        batches = tuple(self.curriculum_pending)
        outcomes = torch.cat(batches)
        self.curriculum_pending.clear()
        if self.curriculum_cohort is None:
            self.curriculum_cohort = torch.zeros_like(batches[0])
            self.curriculum_cohort_seen = torch.zeros(
                len(batches[0]), dtype=torch.bool, device=outcomes.device)
        cohort = self.curriculum_cohort
        seen = self.curriculum_cohort_seen
        assert seen is not None
        # Each environment contributes once; wait for slow failures as well as fast successes.
        for batch in batches:
            selected = batch[:, 1].bool() & batch[:, 3].bool() & ~seen
            cohort[selected] = batch[selected]
            seen |= selected
        complete = bool(seen.all())
        shapes = tuple(self.noise_curriculum.windows)
        recent = {}
        logs = {}
        for index, shape in enumerate(shapes):
            ended = (outcomes[:, 0] == index) & outcomes[:, 1].bool()
            completed = outcomes[ended, 2].bool()
            logs[f"success/{shape}"] = (completed.float().mean() if completed.numel()
                                        else outcomes.new_tensor(0., dtype=torch.float32))
            eligible = (cohort[:, 0] == index) & seen
            recent[shape] = cohort[eligible, 2].bool().cpu().tolist() if complete else []
        previous_scale = self.noise_curriculum.scale
        previous_clearance = self.noise_curriculum.clearance
        previous_difficulty = self.noise_curriculum.difficulty
        upgraded = self.noise_curriculum.update(recent) if complete else False
        if complete:
            seen.zero_()
        if upgraded:
            logging.info(
                "Success curriculum level %d: clearance %.2f -> %.2f mm, "
                "difficulty %.2f -> %.2f, noise %.1f -> %.1f; "
                "all shapes reached 80%% over 1000 episodes.",
                self.noise_curriculum.level, previous_clearance * 1000,
                self.noise_curriculum.clearance * 1000, previous_difficulty,
                self.noise_curriculum.difficulty, previous_scale, self.noise_curriculum.scale,
            )
        for group in self.groups:
            if self.noise_curriculum.noise_levels > 0:
                group.noise_scale = self.noise_curriculum.scale
            group.difficulty = self.noise_curriculum.difficulty
            if hasattr(group, "set_clearance"):
                group.set_clearance(self.noise_curriculum.clearance)
        logs["curriculum/clearance_mm"] = outcomes.new_tensor(
            self.noise_curriculum.clearance * 1000, dtype=torch.float32)
        logs["curriculum/difficulty"] = outcomes.new_tensor(
            self.noise_curriculum.difficulty, dtype=torch.float32)
        logs["curriculum/noise_scale"] = outcomes.new_tensor(
            self.noise_curriculum.scale, dtype=torch.float32)
        for shape, window in self.noise_curriculum.windows.items():
            logs[f"curriculum/episodes/{shape}"] = outcomes.new_tensor(len(window))
            logs[f"curriculum/success/{shape}"] = outcomes.new_tensor(
                sum(window)/len(window) if window else 0., dtype=torch.float32)
        return logs

    def get_observations(self) -> TensorDict:
        pairs = [g.observations() for g in self.groups]
        return TensorDict({"actor": torch.cat([p[0].to(self.device) for p in pairs]),
                           "critic": torch.cat([p[1].to(self.device) for p in pairs])},
                          [self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, 6):
            raise ValueError(f"expected {(self.num_envs, 6)} action, got {actions.shape}")
        results = [g.step(a.to(g.device)) for g, a in zip(
            self.groups, actions.split([g.n for g in self.groups]), strict=True)]
        self.policy_steps += 1
        self.total_transitions += self.num_envs
        for g in self.groups:
            g.noise_scale = (self.noise_curriculum.scale
                             if self.noise_curriculum is not None and self.noise_curriculum.noise_levels > 0
                             else g.cfg.noise_scale(self.policy_steps))
            if g.cfg.training:
                g.difficulty = (self.noise_curriculum.difficulty
                                if self.noise_curriculum is not None
                                else min(1., 0.25 + 0.75 * self.policy_steps / g.cfg.curriculum_steps))
        rewards = torch.cat([r[0].to(self.device) for r in results])
        dones = torch.cat([r[1].to(self.device) for r in results])
        terminal = {key: torch.cat([r[2][key].to(self.device) for r in results])
                    for key in results[0][2]}
        if self.noise_curriculum is not None:
            episode_scales = torch.cat([g.episode_noise_scale.to(self.device) for g in self.groups])
            episode_clearances = torch.cat([g.episode_clearance.to(self.device) for g in self.groups])
            scale_target = (torch.as_tensor(self.noise_curriculum.scale, device=self.device)
                            if self.noise_curriculum.noise_levels > 0 else episode_scales)
            terminal["curriculum_eligible"] = (
                torch.isclose(episode_scales, scale_target)
                & torch.isclose(episode_clearances,
                                torch.as_tensor(self.noise_curriculum.clearance, device=self.device),
                                rtol=0., atol=1e-12))
            if not self.curriculum_distributed:
                self.record_curriculum_outcomes(torch.stack((
                    self.curriculum_shape_ids, dones, terminal["reason"] == 1,
                    terminal["curriculum_eligible"]), dim=1))
        logs = {}
        logs["curriculum/noise_scale"] = torch.tensor(
            self.groups[0].noise_scale, device=self.device)
        for key in self.groups[0].last_reward_terms:
            logs[f"reward/{key}"] = torch.stack(
                [g.last_reward_terms[key].mean().to(self.device)
                 for g in self.groups]).mean()
        successes = {}
        for group, (_, done, info) in zip(self.groups, results, strict=True):
            ids = done.nonzero().flatten()
            if len(ids):
                successes.setdefault(group.shape, []).append(
                    torch.stack(((info["reason"][ids] == 1).float().sum(),
                                 done.float().sum())).to(self.device))
                group.reset(ids)
        for shape, counts in successes.items():
            totals = torch.stack(counts).sum(dim=0)
            if self.noise_curriculum is None:
                logs[f"success/{shape}"] = totals[0]/totals[1]
        self.episode_length_buf.copy_(torch.cat([g.length.to(self.device) for g in self.groups]))
        if self.noise_curriculum is not None:
            logs.pop("curriculum/noise_scale")
            if not self.curriculum_distributed and self.policy_steps % self.curriculum_rollout_steps == 0:
                logs.update(self.advance_noise_curriculum())
        # 超时视为真实任务失败，不向 PPO 发送超时价值自举信号
        return self.get_observations(), rewards, dones, {
            "time_outs": torch.zeros_like(terminal["time_outs"]),
            "log": logs, "terminal": terminal}
