"""基于 mjlab 批量 MuJoCo 仿真的六自由度插孔动力学环境"""

from contextlib import contextmanager, nullcontext
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
from .curriculum import ADR_BOUNDARY_FRACTION, ADRCurriculum, SuccessNoiseCurriculum
from .geometry import build_model, geometry, rough_surface_tiles
from .shapes.base import PARKED_OFFSET_M
from .transforms import (action_target, actor_observation, canonical_quat,
                         quat_inv, quat_mul, quat_rotvec, rotate,
                         rotvec_quat, tip_pose, transform_wrench)


# torch 缓存分配器记录内存块所属的流，Warp 流销毁后再回收会访问失效句柄，故进程内常驻复用
_BLOCKING_STREAMS: dict[tuple[str, int], torch.cuda.ExternalStream] = {}
# Warp 包装外部句柄的对象析构时会注销该句柄，已由 Warp 持有的流须复用原对象，否则原流对 Warp 失效
_WARP_STREAMS: dict[int, wp.Stream] = {}


def blocking_stream(device: str, index: int) -> torch.cuda.ExternalStream:
    key = (device, index)
    if key not in _BLOCKING_STREAMS:
        stream = wp.Stream(device)
        _WARP_STREAMS[stream.cuda_stream] = stream
        _BLOCKING_STREAMS[key] = wp.stream_to_torch(stream)
    return _BLOCKING_STREAMS[key]


def warp_stream(stream: torch.cuda.Stream) -> wp.Stream:
    known = _WARP_STREAMS.get(stream.cuda_stream)
    return wp.stream_from_torch(stream) if known is None else known


def use_warp_stream_in_torch(device: str) -> None:
    # torch 默认的空流上任何操作（含事件记录）都会与全部阻塞组流同步，使组流退化为串行；
    # 主流改用 Warp 设备流（阻塞、非默认），torch 与 warp 同序，空流只剩 Warp 的异步释放
    wp.init()
    stream = wp.get_stream(device)
    _WARP_STREAMS[stream.cuda_stream] = stream
    torch.cuda.set_stream(wp.stream_to_torch(stream))


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


CONTACT_THRESHOLD_N = 0.05
SIMULATION_FAILURE = 5
LOWPASS_OFF_MAX_RAD_S = 100.0


def lowpass_velocity(velocity: torch.Tensor, error: torch.Tensor,
                     omega: float, dt: float) -> torch.Tensor:
    # 临界阻尼二阶目标低通的半隐式欧拉速度更新，随后由调用方用新速度推进状态
    return velocity + dt * (omega * omega * error - 2 * omega * velocity)


def play_hysteresis_step(state: torch.Tensor, signal: torch.Tensor,
                         half_width: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(state, signal-half_width), signal+half_width)


def completion_cost_terms(
    cfg: InsertionCfg, xy: torch.Tensor, z_disp: torch.Tensor,
    target_depth: torch.Tensor, radius: float | torch.Tensor, tilt: torch.Tensor, yaw: torch.Tensor,
    wrench: torch.Tensor, actions: torch.Tensor, previous_actions: torch.Tensor,
    success: torch.Tensor, safety_failure: torch.Tensor, timeout: torch.Tensor,
    *, tip_center_z: torch.Tensor | None = None,
    hole_force: torch.Tensor | None = None,
    stagnation: torch.Tensor | None = None,
    in_contact: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    # 奖励 = -dt * Σ 运行成本 + 一次性终端项；各项定义见 REWARD.md
    dt = cfg.physics_dt*cfg.decimation
    distance = position_distance(cfg, xy, z_disp, target_depth, radius)
    orientation = orientation_cost(cfg, z_disp, target_depth, tilt, yaw)
    force, moment = load_costs(cfg, wrench, tip_center_z, hole_force)
    magnitude, change = action_costs(cfg, actions, previous_actions, in_contact)
    if stagnation is None:
        stagnation = torch.zeros_like(xy)
    return {
        "time_penalty": torch.full_like(xy, -dt*cfg.time_cost_per_second),
        "position_penalty": -dt*cfg.position_cost_per_second*distance,
        "orientation_penalty": -dt*cfg.orientation_cost_per_second*orientation,
        "force_penalty": -dt*cfg.force_cost_per_second*force.square(),
        "moment_penalty": -dt*cfg.moment_cost_per_second*moment.square(),
        "stagnation_penalty": -dt*cfg.stagnation_cost_per_second*stagnation,
        "action_penalty": -dt*cfg.action_cost_per_second*magnitude,
        "action_change_penalty": -dt*change,
        "completion_bonus": cfg.completion_bonus*success.float(),
        "safety_failure_penalty": -cfg.safety_failure_penalty*safety_failure.float(),
        "timeout_penalty": -cfg.timeout_penalty*(timeout & ~success & ~safety_failure).float(),
    }


def position_distance(cfg: InsertionCfg, xy: torch.Tensor, z_disp: torch.Tensor,
                      target_depth: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """D = D_xy + D_z，两项各自饱和于 position_cost_cap/2。"""
    half_cap = 0.5*cfg.position_cost_cap
    if cfg.surface_z_free_m > 0:
        # 板面上方 surface_z_free_m 内的抬升不增加深度成本（第二阶段越障）
        lowest_z = z_disp - target_depth
        z_disp = (target_depth + lowest_z.clamp_max(0.0)
                  + (lowest_z - cfg.surface_z_free_m).clamp_min(0.0))
    remaining = z_disp.clamp_min(0.0)/target_depth
    xy_distance = half_cap*torch.tanh(
        (xy/(radius*cfg.xy_cost_radius_fraction)).square()/half_cap)
    w = cfg.xy_outer_cost_weight
    if w > 0:
        # 远场有界径向斜率，消除近场 tanh 饱和后的零梯度
        eps, r_max = 0.05*radius, cfg.workspace_xy_radius_m
        x2 = xy.clamp(0.0, r_max).square()
        length = (math.hypot(r_max, eps) if isinstance(eps, float)
                  else torch.sqrt(r_max*r_max + eps.square()))
        norm = r_max*r_max / (length + eps)
        outer_xy = half_cap * x2 / ((torch.sqrt(x2 + eps*eps) + eps) * norm)
        xy_distance = (1.0 - w)*xy_distance + w*outer_xy
    z_distance = half_cap*torch.tanh(remaining/half_cap)
    w = cfg.z_outer_cost_weight
    if w > 0:
        # 板面到工作空间顶部线性增长，消除离板后 tanh 饱和的零梯度（否则悬停高空与贴板同价）
        ceiling = cfg.workspace_z_max_m*cfg.scale
        outer_z = half_cap*(z_disp - target_depth).clamp(0.0, ceiling)/ceiling
        z_distance = (1.0 - w)*z_distance + w*outer_z
    return xy_distance + z_distance


def orientation_cost(cfg: InsertionCfg, z_disp: torch.Tensor, target_depth: torch.Tensor,
                     tilt: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """O = p·h(倾角) + h(偏航)，p 为最低点入孔前 4 mm 的线性进度。"""
    reference = math.radians(cfg.orientation_cost_reference_deg)
    tilt_error, yaw_error = (tilt/reference).square(), (yaw/reference).square()
    tilt_weight = ((target_depth-z_disp)/0.004).clamp(0, 1)
    return tilt_weight*tilt_error/(1+tilt_error) + yaw_error/(1+yaw_error)


def load_costs(cfg: InsertionCfg, wrench: torch.Tensor, tip_center_z: torch.Tensor | None,
               hole_force: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """归一化超死区力/力矩，上限 load_cost_cap；入孔后渐撤孔系 Fz 与死区。"""
    cap = cfg.load_cost_cap
    moment_norm = torch.linalg.vector_norm(wrench[:, 3:], dim=-1)
    if cfg.inserted_load_cost and tip_center_z is not None and hole_force is not None:
        z_weight = (1.0 + tip_center_z / cfg.inserted_load_depth_m).clamp(0.0, 1.0)
        force_norm = torch.linalg.vector_norm(torch.cat((
            hole_force[:, :2], (hole_force[:, 2] * z_weight).unsqueeze(-1)), dim=-1), dim=-1)
    else:
        z_weight = torch.ones_like(moment_norm)
        force_norm = torch.linalg.vector_norm(wrench[:, :3], dim=-1)
    free_f = cfg.force_cost_free_n * z_weight
    free_m = cfg.moment_cost_free_nm * z_weight
    force = ((force_norm - free_f) / (cfg.max_force - free_f)).clamp(0, cap)
    moment = ((moment_norm - free_m) / (cfg.max_moment - free_m)).clamp(0, cap)
    return force, moment


def stagnation_level(cfg: InsertionCfg, tip: torch.Tensor, tip_ema: torch.Tensor,
                     length: torch.Tensor, goal_z: torch.Tensor) -> torch.Tensor:
    """未达目标深度时净推进速率低于阈值的平方缺口；原地更新 tip_ema。"""
    dt = cfg.physics_dt * cfg.decimation
    if cfg.rough_surface:
        # 只统计横向与向下进度，板面上方上下弹跳不算推进
        tip = torch.cat((tip[:, :2], tip[:, 2:3].clamp_max(0.0)), dim=-1)
    delta = tip - tip_ema
    speed = torch.linalg.vector_norm(delta, dim=-1) / cfg.stagnation_window_seconds
    tip_ema.add_((1.0 - math.exp(-dt / cfg.stagnation_window_seconds)) * delta)
    warmup = (length.to(tip.dtype) * dt / cfg.stagnation_window_seconds).clamp(0.0, 1.0)
    not_at_goal = (tip[:, 2] > goal_z + 0.0005).to(tip.dtype)
    return warmup * not_at_goal * (1.0 - speed / cfg.stagnation_speed_threshold_mps).clamp(0.0, 1.0).square()


def action_costs(cfg: InsertionCfg, actions: torch.Tensor, previous_actions: torch.Tensor,
                 in_contact: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 ‖ā‖ 与 w_Δa·‖ā−ā_prev‖（悬空可选平方）；ā 为 tanh 软饱和动作，悬空时 w_Δa 更大。"""
    applied = cfg.action_cost_cap * torch.tanh(actions / cfg.action_cost_cap)
    previous = cfg.action_cost_cap * torch.tanh(previous_actions / cfg.action_cost_cap)
    change = torch.linalg.vector_norm(applied-previous, dim=-1)
    cost = cfg.action_change_cost_per_second*change
    if in_contact is not None and cfg.free_air_action_change_cost_per_second is not None:
        air = change.square() if cfg.free_air_action_change_squared else change
        cost = torch.where(in_contact, cost, cfg.free_air_action_change_cost_per_second*air)
    return torch.linalg.vector_norm(applied, dim=-1), cost


class PlantGroup:
    def __init__(self, shape: str, cfg: InsertionCfg, device: str):
        self.cfg, self.device, self.n = cfg, device, cfg.envs_per_shape
        self.last_reward_terms: dict[str, torch.Tensor] = {}
        self.adr = cfg.training and cfg.noise_curriculum and cfg.noise_curriculum_mode == "adr"
        # ADR 按目标间隙建模一次，回合间隙在 [clearance_floor, clearance_max] 内逐环境变化
        self.clearance_max = cfg.active_clearance_levels[0]
        self.clearance = cfg.clearance if self.adr else self.clearance_max
        self.clearance_floor = self.clearance_max
        self.shape = shape
        self._build_simulation()
        self._init_shape()
        self.shape_params = torch.tensor((*self.geometry.definition.nominal_params(self.geometry), self.clearance),
                                         device=device).expand(self.n, -1).clone()
        self.rim_height = torch.zeros(self.n, device=device)
        self.rim_width = torch.zeros(self.n, device=device)
        self.opening_radius = self.geometry.definition.opening_radius(self.geo, self.shape_params).clone()
        self.generator = torch.Generator(device=device).manual_seed(
            cfg.seed + ALL_SHAPES.index(shape) * 10007)
        self.difficulty = 0.0 if cfg.training else 1.0
        self.constants: dict[tuple[float, ...], torch.Tensor] = {}
        self.noise_scale = cfg.noise_scale(0)
        # ADR 收拢阶段的回合噪声系数下限
        self.noise_floor = 0.0
        self.episode_noise_scale = torch.full((self.n,), self.noise_scale, device=device)
        # ADR 边界回合钉住的维度：0 普通回合，1 噪声，2 间隙
        self.episode_boundary = torch.zeros(self.n, dtype=torch.long, device=device)
        self.nominal_offset = torch.tensor([0, 0, -0.05 * cfg.scale],
                                           device=device).expand(self.n, 3)
        self.offset = self.nominal_offset.clone()
        self.sensor_offset = torch.tensor([0, 0, 0.03 * cfg.scale],
                                          device=device).repeat(self.n, 1)
        self.com_offset = torch.zeros(self.n, 3, device=device)
        self.identity = torch.tensor([1., 0, 0, 0], device=device).expand(self.n, 4)
        self.goal = torch.zeros(self.n, 3, device=device)
        self.length = torch.zeros(self.n, device=device, dtype=torch.long)
        self.position_delay_frames = torch.zeros_like(self.length)
        self.position_history = torch.zeros(self.n, 2, 3, device=device)
        self.position_history_length = torch.full_like(self.length, -1)
        self.hold = torch.zeros_like(self.length)
        self.peak_force = torch.zeros(self.n, device=device)
        self.peak_moment = torch.zeros_like(self.peak_force)
        self.step_peak_force = torch.zeros_like(self.peak_force)
        self.step_peak_moment = torch.zeros_like(self.peak_force)
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
        self.stiction = torch.zeros_like(self.k)
        self.wall_friction = torch.zeros(self.n, device=device)
        self.wall_force_z = torch.zeros(self.n, device=device)
        self.surface_max_height = torch.zeros(self.n, device=device)
        self.tip_ema = torch.zeros(self.n, 3, device=device)
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
        self.apparent_mass = torch.ones(self.n, 1, device=device)
        self.inertia = torch.zeros(self.n, 3, device=device)
        self.down_mass = torch.zeros_like(self.mass)
        self.down_inertia = torch.zeros_like(self.inertia)
        self.calibration = torch.zeros(self.n, 3, device=device)
        self.sensor_error = self.identity.clone()
        self.rotation_bias_q = self.identity.clone()
        self.bias = torch.zeros(self.n, 3, device=device)
        self.delay = torch.zeros(self.n, device=device, dtype=torch.long)
        self.targets_p = torch.zeros(3, self.n, 3, device=device)
        self.targets_q = self.identity.unsqueeze(0).repeat(3, 1, 1)
        self.physics_p_goal = torch.zeros(self.n, 3, device=device)
        self.physics_q_goal = self.identity.clone()
        self.lowpass_p = torch.zeros(self.n, 3, device=device)
        self.lowpass_v = torch.zeros_like(self.lowpass_p)
        self.lowpass_q = self.identity.clone()
        self.lowpass_w = torch.zeros_like(self.lowpass_p)
        self.lowpass_off = torch.zeros(self.n, 1, device=device, dtype=torch.bool)
        self.lowpass_omega = torch.full_like(self.lowpass_p, cfg.target_lowpass_translation_rad_s)
        self.lowpass_zeta = torch.ones_like(self.lowpass_p)
        self.pivot = torch.zeros(self.n, 3, device=device)
        self.slip_amplitude = torch.zeros(self.n, 3, device=device)
        self.slip_omega = torch.zeros(self.n, device=device)
        self.slip_phase = torch.zeros(self.n, device=device)
        self.clock = torch.zeros(self.n, device=device)
        self.step_bias_active = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.step_peak_contact = torch.zeros(self.n, device=device)
        self.step_contact_overflow = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.hard_capacity = 4096
        self.hard_pos = torch.zeros(self.hard_capacity, 3, device=device)
        self.hard_q = torch.tensor([1., 0, 0, 0], device=device).repeat(self.hard_capacity, 1)
        self.hard_valid = torch.zeros(self.hard_capacity, device=device, dtype=torch.bool)
        self.hard_params = self.shape_params[:1].expand(self.hard_capacity, -1).clone()
        self.hard_rim = torch.zeros(self.hard_capacity, 2, device=device)
        self.hard_slot = 0
        self.hard_start = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.vibration = torch.zeros(self.n, 6, device=device)
        self.vibration_state = torch.zeros(self.n, 2, 6, device=device)
        self.vibration_phi = torch.zeros(self.n, 2, 6, device=device)
        self.vibration_gain = torch.zeros(self.n, 6, device=device)
        self.vibration_amplitude = torch.zeros(self.n, 1, device=device)
        # 使用张量视图避免 TorchArray 索引写入触发跨流切换，以便被 CUDA Graph 捕获
        self.applied_wrench = self.sim.data.xfrc_applied[:, self.body_id]
        self.physics_graph: torch.cuda.CUDAGraph | None = None
        self.sensor_graph: torch.cuda.CUDAGraph | None = None
        self.physics_rollout_graph: torch.cuda.CUDAGraph | None = None
        self.reset_constants_graph: torch.cuda.CUDAGraph | None = None
        self.masked_reset_graph: torch.cuda.CUDAGraph | None = None
        self.masked_reset_signature: tuple[float, ...] | None = None
        self.masked_reset_candidate: tuple[float, ...] | None = None
        self.reset_request = torch.zeros(self.n, device=device, dtype=torch.bool)
        self.everyone = torch.arange(self.n, device=device)
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
                                 cfg.plate_outer_radius_m, cfg.rough_surface, cfg.hole_depth_m,
                                 cfg.shape_randomization)
        self.model = build_model(self.geometry)
        self.body_id = self.model.body("tool").id
        nconmax, njmax = self.geometry.definition.contact_limits(cfg.rough_surface)
        self.sim = Simulation(
            self.n, SimulationCfg(nconmax=nconmax, njmax=njmax,
                                 mujoco=MujocoCfg(timestep=cfg.physics_dt, gravity=(0, 0, 0),
                                                 iterations=50, ls_iterations=20)),
            model=self.model, device=self.device,
        )
        expanded = (("body_mass", "body_inertia", "geom_friction", "geom_pos", "geom_quat")
                    if cfg.rough_surface
                    else ("body_mass", "body_inertia", "geom_friction", "geom_pos"))
        if cfg.shape_randomization:
            expanded += ("geom_size", "geom_rbound", "geom_aabb")
        self.sim.expand_model_fields(expanded)
        self.expanded_fields = expanded
        self.nefc_view = wp.to_torch(self.sim.wp_data.nefc)
        self.nacon_view = wp.to_torch(self.sim.wp_data.nacon)
        self.njmax, self.naconmax = int(self.sim.wp_data.njmax), int(self.sim.wp_data.naconmax)
        self.peg_geom_ids = (self.model.geom_bodyid == self.body_id).nonzero()[0].tolist()
        self.default_geom_pos = torch.tensor(
            self.model.geom_pos, device=self.device, dtype=torch.float32)
        self.peg_polygon = torch.tensor(
            self.geometry.peg_polygon, dtype=torch.float32, device=self.device)
        self.hole_polygon = torch.tensor(
            self.geometry.hole_polygon, dtype=torch.float32, device=self.device)
        self._init_rough_surface()

    def _init_shape(self) -> None:
        self.geo = self.geometry.definition.prepare(self.geometry, self.device)
        names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        self.geom_index = {name: i for i, name in enumerate(names)}
        hole_body = self.model.body("hole").id
        self.rim_geom_ids = [i for i, name in enumerate(names) if name.startswith("rim_")]
        self.plate_geom_ids = [i for i in range(self.model.ngeom)
                               if self.model.geom_bodyid[i] == hole_body and i not in self.rim_geom_ids]
        self.geom_base_pos = self.default_geom_pos.expand(self.n, -1, -1).clone()
        if self.cfg.shape_randomization:
            self.default_geom_size = torch.tensor(self.model.geom_size, device=self.device, dtype=torch.float32)

    def geom_ids_with_prefix(self, prefix: str) -> list[int]:
        return [i for name, i in self.geom_index.items() if name.startswith(prefix)]

    def write_box(self, ids: torch.Tensor, geom_id: int, half: torch.Tensor) -> None:
        # 改写盒体尺寸时同步宽相包围半径与局部包围盒，否则宽相碰撞会漏检
        # 仿真字段均经 [:] 视图在当前流上写入，TorchArray 索引写入会切到 warp 流而无法被图捕获
        self.sim.model.geom_size[:][ids, geom_id] = half
        self.sim.model.geom_rbound[:][ids, geom_id] = torch.linalg.vector_norm(half, dim=-1)
        self.sim.model.geom_aabb[:][ids, geom_id, 1] = half

    def set_box(self, ids: torch.Tensor, name: str, center_xy: torch.Tensor, half_xy: torch.Tensor) -> None:
        """按环境改写盒体的 XY 中心与半尺寸，Z 方向保持建模值"""
        geom_id = self.geom_index[name]
        self.geom_base_pos[ids, geom_id, :2] = center_xy.float()
        z_half = self.default_geom_size[geom_id, 2:].expand(len(ids), 1)
        self.write_box(ids, geom_id, torch.cat((half_xy.float(), z_half), -1))

    def randomize_shape(self, ids: torch.Tensor) -> None:
        cfg, n = self.cfg, len(ids)
        self.shape_params[ids, :3] = self.geometry.definition.sample_params(self.geometry, n, self.uniform).float()
        present = self.uniform(n, 1, 0., 1.)[:, 0] < cfg.rim_probability * self.difficulty
        height = self.curriculum_uniform(
            n, 1, cfg.rim_height_min_m, cfg.rim_height_max_m, cfg.rim_height_min_m)[:, 0] * cfg.scale
        self.rim_height[ids] = torch.where(present, height, torch.zeros_like(height))
        self.rim_width[ids] = self.uniform(n, 1, cfg.rim_width_min_m, cfg.rim_width_max_m)[:, 0] * cfg.scale
        self.apply_shape(ids)

    def apply_shape(self, ids: torch.Tensor) -> None:
        definition, params = self.geometry.definition, self.shape_params[ids]
        definition.apply_params(self, ids, params)
        self.opening_radius[ids] = definition.opening_radius(self.geo, params)
        if not self.rim_geom_ids:
            return
        centers, half = definition.rim_boxes(self.geo, params, self.rim_width[ids])
        height = self.rim_height[ids, None]
        present = height > 0
        # 围框顶面为孔口 z=0；无围框的环境把围框盒体移到远处
        z = torch.where(present, -height/2, torch.full_like(height, -PARKED_OFFSET_M))
        z_half = torch.where(present, height/2, torch.full_like(height, 1e-3))
        for b, geom_id in enumerate(self.rim_geom_ids):
            self.geom_base_pos[ids, geom_id] = torch.cat((centers[:, b].float(), z), -1)
            self.write_box(ids, geom_id, torch.cat((half[:, b].float(), z_half), -1))

    def apply_geom_pos(self, ids: torch.Tensor, grasp: torch.Tensor) -> None:
        base = self.geom_base_pos[ids]
        geom_pos = self.sim.model.geom_pos[:]
        for geom_id in self.peg_geom_ids:
            geom_pos[ids, geom_id] = base[:, geom_id] + grasp
        if not self.cfg.shape_randomization:
            return
        lowered = torch.zeros_like(grasp)
        lowered[:, 2] = self.rim_height[ids]
        for geom_id in self.plate_geom_ids:
            geom_pos[ids, geom_id] = base[:, geom_id] - lowered
        for geom_id in self.rim_geom_ids:
            geom_pos[ids, geom_id] = base[:, geom_id]

    def _init_rough_surface(self) -> None:
        tiles = rough_surface_tiles(self.geometry)
        if not tiles:
            self.surf_geom_ids: list[int] = []
            self.surf_slice: slice | None = None
            self.surf_tile_top_z: torch.Tensor | None = None
            self.surf_tile_xy: torch.Tensor | None = None
            self.surf_horiz_radius: torch.Tensor | None = None
            return
        surface_body_id = self.model.body("surface").id
        self.surf_geom_ids = (self.model.geom_bodyid == surface_body_id).nonzero()[0].tolist()
        expected = list(range(self.surf_geom_ids[0], self.surf_geom_ids[-1] + 1))
        if self.surf_geom_ids != expected or len(self.surf_geom_ids) != len(tiles):
            raise RuntimeError("rough surface geoms must form a contiguous slice matching tile specs")
        self.surf_slice = slice(self.surf_geom_ids[0], self.surf_geom_ids[-1] + 1)
        self.surf_xy = torch.tensor(
            [[t.cx, t.cy] for t in tiles], device=self.device, dtype=torch.float32)
        self.surf_sizes = torch.tensor(
            [[t.sx, t.sy, t.sz] for t in tiles], device=self.device, dtype=torch.float32)
        self.surf_horiz_radius = torch.hypot(self.surf_sizes[:, 0], self.surf_sizes[:, 1])
        self.surf_tile_top_z = torch.zeros(self.n, len(tiles), device=self.device, dtype=torch.float32)
        self.surf_tile_xy = self.surf_xy.unsqueeze(0).expand(self.n, -1, -1).clone()
        yaws = torch.tensor([t.yaw for t in tiles], device=self.device, dtype=torch.float32)
        self.surf_q_yaw = torch.stack(
            (torch.cos(yaws / 2), torch.zeros_like(yaws),
             torch.zeros_like(yaws), torch.sin(yaws / 2)), dim=-1)
        u_rows, b_rows, act_rows = [], [], []
        for c_idx in range(2):
            u_c, b_c, a_c = [], [], []
            for t in tiles:
                if c_idx < len(t.guards):
                    gx, gy, gb = t.guards[c_idx]
                    u_c.append([gx, gy, 0.0])
                    b_c.append(gb)
                    a_c.append(1.0)
                else:
                    u_c.append([1.0, 0.0, 0.0])
                    b_c.append(-1e9)
                    a_c.append(0.0)
            u_rows.append(u_c)
            b_rows.append(b_c)
            act_rows.append(a_c)
        self.surf_guard_u = torch.tensor(u_rows, device=self.device, dtype=torch.float32)
        self.surf_guard_b = torch.tensor(b_rows, device=self.device, dtype=torch.float32)
        self.surf_guard_active = torch.tensor(act_rows, device=self.device, dtype=torch.float32)
        inner_flags = [
            min(t.cx * gx + t.cy * gy - gb for gx, gy, gb in t.guards) < 0.0032 * self.cfg.scale
            for t in tiles
        ]
        self.surf_inner_ring = torch.tensor(inner_flags, device=self.device, dtype=torch.bool)
        self.surf_ex = torch.tensor([[1., 0., 0.]], device=self.device, dtype=torch.float32)
        self.surf_ey = torch.tensor([[0., 1., 0.]], device=self.device, dtype=torch.float32)

    def _compute_rough_tiles(
        self, amp: torch.Tensor, wavelengths: torch.Tensor, theta: torch.Tensor,
        phase: torch.Tensor, weights: torch.Tensor,
        jitter_u: torch.Tensor | None = None, stepped_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n, k_tiles = amp.shape[0], len(self.surf_geom_ids)
        wnum = 2 * math.pi / wavelengths
        kx, ky = wnum * theta.cos(), wnum * theta.sin()
        arg = (self.surf_xy[None, :, 0:1] * kx[:, None, :]
               + self.surf_xy[None, :, 1:2] * ky[:, None, :]
               + phase[:, None, :])
        psi = (weights[:, None, :] * arg.sin()).sum(dim=-1)
        cos_w = weights[:, None, :] * arg.cos()
        dpsi_dx = (cos_w * kx[:, None, :]).sum(dim=-1)
        dpsi_dy = (cos_w * ky[:, None, :]).sum(dim=-1)

        if cfg.rough_step_jitter > 0 and jitter_u is not None:
            base_span = (psi.amax(dim=-1, keepdim=True) - psi.amin(dim=-1, keepdim=True)).clamp_min(1e-6)
            psi = psi + jitter_u * (cfg.rough_step_jitter * base_span)

        psi_min = psi.amin(dim=-1, keepdim=True)
        scale_h = amp / (psi.amax(dim=-1, keepdim=True) - psi_min).clamp_min(1e-6)
        h = scale_h * (psi - psi_min)
        dh_dx, dh_dy = scale_h * dpsi_dx, scale_h * dpsi_dy
        if stepped_mask is not None:
            dh_dx = torch.where(stepped_mask, torch.zeros_like(dh_dx), dh_dx)
            dh_dy = torch.where(stepped_mask, torch.zeros_like(dh_dy), dh_dy)

        outer_slope = math.tan(math.radians(cfg.rough_max_slope_deg))
        inner_slope = min(outer_slope, math.tan(math.radians(5.0)))
        tile_max_slope = torch.where(
            self.surf_inner_ring[None, :],
            torch.full_like(dh_dx, inner_slope),
            torch.full_like(dh_dx, outer_slope),
        )
        slope_cap = (tile_max_slope / torch.hypot(dh_dx, dh_dy).clamp_min(1e-8)).clamp_max(1.0)
        dh_dx, dh_dy = dh_dx * slope_cap, dh_dy * slope_cap

        n_raw = torch.stack((-dh_dx, -dh_dy, torch.ones_like(h)), dim=-1)
        n_surf = n_raw / torch.linalg.vector_norm(n_raw, dim=-1, keepdim=True)
        sin_b = torch.hypot(n_surf[..., 0], n_surf[..., 1])
        beta = torch.atan2(sin_b, n_surf[..., 2])
        ratio = torch.where(sin_b > 1e-6, beta / sin_b.clamp_min(1e-6), torch.ones_like(sin_b))
        rv_tilt = torch.stack(
            (-n_surf[..., 1] * ratio, n_surf[..., 0] * ratio, torch.zeros_like(beta)), dim=-1)
        q_tilt = rotvec_quat(rv_tilt.reshape(-1, 3)).reshape(n, k_tiles, 4)
        q_full = quat_mul(
            q_tilt.reshape(-1, 4),
            self.surf_q_yaw.unsqueeze(0).expand(n, -1, -1).reshape(-1, 4),
        ).reshape(n, k_tiles, 4)

        top_center = torch.cat(
            (self.surf_xy.unsqueeze(0).expand(n, -1, -1), h.unsqueeze(-1)), dim=-1)
        pos = top_center - self.surf_sizes[None, :, 2:3] * n_surf
        flat_q = q_full.reshape(-1, 4)
        ex = rotate(flat_q, self.surf_ex.expand(n * k_tiles, 3)).reshape(n, k_tiles, 3)
        ey = rotate(flat_q, self.surf_ey.expand(n * k_tiles, 3)).reshape(n, k_tiles, 3)
        ez = n_surf
        for c_idx in range(2):
            u_vec = self.surf_guard_u[c_idx]
            b_val = self.surf_guard_b[c_idx]
            act = self.surf_guard_active[c_idx]
            proj_c = (pos * u_vec[None, :, :]).sum(dim=-1)
            extent = (
                (ex * u_vec[None, :, :]).sum(dim=-1).abs() * self.surf_sizes[None, :, 0]
                + (ey * u_vec[None, :, :]).sum(dim=-1).abs() * self.surf_sizes[None, :, 1]
                + (ez * u_vec[None, :, :]).sum(dim=-1).abs() * self.surf_sizes[None, :, 2]
            )
            deficit = (b_val[None, :] - (proj_c - extent)).clamp_min(0.0) * act[None, :]
            pos = pos + deficit.unsqueeze(-1) * u_vec[None, :, :]

        corner_ext_z = (
            ex[..., 2].abs() * self.surf_sizes[None, :, 0]
            + ey[..., 2].abs() * self.surf_sizes[None, :, 1]
            + ez[..., 2].abs() * self.surf_sizes[None, :, 2]
        )
        tile_top_z = pos[..., 2] + corner_ext_z
        excess_z = (tile_top_z.amax(dim=-1, keepdim=True) - amp).clamp_min(0.0)
        pos = torch.cat((pos[..., :2], pos[..., 2:3] - excess_z.unsqueeze(-1)), dim=-1)
        tile_top_z = tile_top_z - excess_z
        # 当瓦片顶面与 z=0 基板共面时将其下沉到基板内部，消除共面重复接触导致的奇异海森矩阵
        coplanar_bury = (tile_top_z <= 5e-5 * cfg.scale).to(pos.dtype) * (0.010 * cfg.scale)
        pos = torch.cat((pos[..., :2], pos[..., 2:3] - coplanar_bury.unsqueeze(-1)), dim=-1)
        tile_top_z = torch.where(coplanar_bury > 0, torch.zeros_like(tile_top_z), tile_top_z)
        return pos, q_full, tile_top_z

    def _randomize_rough_surface(self, ids: torch.Tensor) -> None:
        if self.surf_slice is None or len(ids) == 0:
            self.surface_max_height[ids] = 0
            return
        cfg = self.cfg
        n, k_tiles, m_waves = len(ids), len(self.surf_geom_ids), cfg.rough_num_waves
        wl_min, wl_max = cfg.rough_wavelength_min_m * cfg.scale, cfg.rough_wavelength_max_m * cfg.scale
        wavelengths = self.uniform(n, m_waves, math.log(wl_min), math.log(wl_max)).exp()
        theta = self.uniform(n, m_waves, -math.pi, math.pi)
        phase = self.uniform(n, m_waves, -math.pi, math.pi)
        weights = self.uniform(n, m_waves, 0.2, 1.0)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        amp_scale = self.difficulty if cfg.training else 1.0
        amp = self.uniform(n, 1, cfg.rough_min_amplitude_m * cfg.scale,
                           cfg.rough_max_amplitude_m * cfg.scale) * amp_scale
        if cfg.rough_flat_probability > 0:
            flat_mask = ((ids < round(self.n * cfg.rough_flat_probability)).unsqueeze(-1)
                         if cfg.training and self.n >= 4
                         else self.uniform(n, 1, 0.0, 1.0) < cfg.rough_flat_probability)
            amp = torch.where(flat_mask, torch.zeros_like(amp), amp)
        jitter = self.uniform(n, k_tiles, -0.5, 0.5) if cfg.rough_step_jitter > 0 else None
        stepped = (self.uniform(n, 1, 0.0, 1.0) < cfg.rough_stepped_prob
                   if cfg.rough_stepped_prob > 0 else None)
        pos, q_full, tile_top_z = self._compute_rough_tiles(
            amp, wavelengths, theta, phase, weights, jitter, stepped)
        self.sim.model.geom_pos[:][ids, self.surf_slice] = pos
        self.sim.model.geom_quat[:][ids, self.surf_slice] = q_full
        if self.surf_tile_top_z is not None and self.surf_tile_xy is not None:
            self.surf_tile_top_z[ids] = tile_top_z
            self.surf_tile_xy[ids] = pos[..., :2]
        self.surface_max_height[ids] = tile_top_z.amax(dim=-1).clamp_min(0.0)

    def uniform(self, n: int, columns: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, columns, generator=self.generator,
                                             device=self.device)

    def constant(self, *values: float) -> torch.Tensor:
        # 重置中复用的设备常量只创建一次，避免每次从分页内存同步拷贝
        tensor = self.constants.get(values)
        if tensor is None:
            tensor = self.constants[values] = torch.tensor(values, device=self.device)
        return tensor

    def curriculum_uniform(self, n: int, columns: int, low: float, high: float,
                           center: float) -> torch.Tensor:
        # 课程难度把采样区间向 center 线性收缩，评估时难度为 1 即全幅
        return center + (self.uniform(n, columns, low, high) - center) * self.difficulty

    def normal(self, n: int, columns: int, std: float) -> torch.Tensor:
        if std == 0:
            return torch.zeros(n, columns, device=self.device)
        return torch.randn(n, columns, generator=self.generator, device=self.device) * std

    def unit_vector(self, n: int) -> torch.Tensor:
        azimuth = self.uniform(n, 1, -math.pi, math.pi)
        z = self.uniform(n, 1, -1., 1.)
        xy = (1-z.square()).clamp_min(0).sqrt()
        return torch.cat((xy*azimuth.cos(), xy*azimuth.sin(), z), dim=-1)

    def sample_wrench_bias(self, n: int, amplitude: float) -> torch.Tensor:
        if self.cfg.wrench_bias_distribution == "per_axis":
            return self.uniform(n, 3, -amplitude, amplitude)
        return PlantGroup.unit_vector(self, n)*self.uniform(n, 1, 0., amplitude)

    def record_hard_states(self) -> None:
        # 每步抽查少量环境，把持续接触且未到目标的真实尖端位姿写入环形缓冲，全程无主机同步
        k = min(8, self.n)
        idx = torch.randint(0, self.n, (k,), generator=self.generator, device=self.device)
        pos, q, _, _ = self.state()
        tip = tip_pose(pos[idx], q[idx], self.offset[idx])
        tip_z = tip[:, 2]
        if self.cfg.shape_randomization:
            # 围框外低处平面上的卡滞状态同样算作孔外
            inside = self.geometry.definition.opening_contains(
                self.geo, self.shape_params[idx], tip[:, None, :2])[:, 0]
            tip_z = torch.where(inside, tip_z, tip_z.clamp_min(0.))
        # 已入孔超过 1 mm 的状态几乎必成功；只记录挣扎 ≥2 s 仍在板面或孔口的接触状态
        eligible = ((self.step_peak_contact[idx] > CONTACT_THRESHOLD_N) & (self.length[idx] >= 100)
                    & (tip_z > -0.001 * self.cfg.scale))
        slots = (self.hard_slot + torch.arange(k, device=self.device)) % self.hard_capacity
        self.hard_slot = (self.hard_slot + k) % self.hard_capacity
        keep = eligible.unsqueeze(-1)
        self.hard_pos[slots] = torch.where(keep, tip, self.hard_pos[slots])
        self.hard_q[slots] = torch.where(keep, q[idx], self.hard_q[slots])
        # 位姿只在当时的孔形尺寸下有效，回放时一并恢复
        self.hard_params[slots] = torch.where(keep, self.shape_params[idx], self.hard_params[slots])
        self.hard_rim[slots] = torch.where(keep, torch.stack((self.rim_height[idx], self.rim_width[idx]), -1),
                                           self.hard_rim[slots])
        self.hard_valid[slots] = self.hard_valid[slots] | eligible

    def sample_hard_start(self, ids: torch.Tensor, pos: torch.Tensor,
                          q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg, n = self.cfg, len(ids)
        self.hard_start.index_fill_(0, ids, False)
        if not cfg.training or cfg.hard_state_reset_probability == 0:
            return pos, q
        slot = torch.multinomial(self.hard_valid.float() + 1e-12, n, replacement=True,
                                 generator=self.generator)
        hard_pos, hard_q = self.hard_pos[slot].clone(), self.hard_q[slot]
        use = ((self.uniform(n, 1, 0., 1.)[:, 0] < cfg.hard_state_reset_probability * self.difficulty)
               & self.hard_valid[slot] & (hard_pos[:, 2] > self.goal[ids, 2] + 0.002 * cfg.scale))
        if cfg.rough_surface:
            # 新回合地形不同，孔外位姿抬高到不穿透新地形；孔内位姿不受地形影响
            inside = (hard_pos[:, 2] < 0) & self.geometry.definition.opening_contains(
                self.geo, self.shape_params[ids], hard_pos[:, None, :2])[:, 0]
            clear_z = self.initial_minimum_tip_height(
                hard_q, margin=0.0005 * cfg.scale, surface_height=self.surface_max_height[ids], ids=ids)
            hard_pos[:, 2] = torch.where(inside, hard_pos[:, 2], torch.maximum(hard_pos[:, 2], clear_z))
        if cfg.shape_randomization:
            keep = use.unsqueeze(-1)
            self.shape_params[ids] = torch.where(keep, self.hard_params[slot], self.shape_params[ids])
            rim = torch.where(keep, self.hard_rim[slot],
                              torch.stack((self.rim_height[ids], self.rim_width[ids]), -1))
            self.rim_height[ids], self.rim_width[ids] = rim[:, 0], rim[:, 1]
            self.apply_shape(ids)
            self.apply_geom_pos(ids, self.offset[ids] - self.nominal_offset[ids])
        self.hard_start[ids] = use
        return (torch.where(use.unsqueeze(-1), hard_pos, pos),
                torch.where(use.unsqueeze(-1), hard_q, q))

    def reset_vibration(self, ids: torch.Tensor) -> None:
        # 窄带振动用 AR(2) 谐振器按策略周期离散，增益归一化使稳态标准差为 1
        cfg, n = self.cfg, len(ids)
        dt = cfg.physics_dt * cfg.decimation
        omega = 2 * math.pi * self.uniform(n, 6, cfg.vibration_min_hz, cfg.vibration_max_hz)
        zeta = self.uniform(n, 6, 0.1, 0.3)
        amplitude = (self.uniform(n, 1, cfg.vibration_force_min_n, cfg.vibration_force_max_n)
                     * self.episode_noise_scale[ids, None])
        if self.cfg.dynamics == "real":
            omega.fill_(2 * math.pi * 6.0)
            zeta.fill_(0.2)
            amplitude.fill_(1.0)
        elif self.cfg.dynamics == "nominal":
            amplitude.zero_()
        radius = torch.exp(-zeta * omega * dt)
        phi1 = 2 * radius * torch.cos(omega * dt * torch.sqrt(1 - zeta.square()))
        phi2 = -radius.square()
        variance = (1 - phi2) / ((1 + phi2) * ((1 - phi2).square() - phi1.square()))
        self.vibration_phi[ids] = torch.stack((phi1, phi2), 1)
        self.vibration_gain[ids] = variance.rsqrt()
        self.vibration_amplitude[ids] = amplitude
        self.vibration_state.index_fill_(0, ids, 0)
        self.vibration.index_fill_(0, ids, 0)

    def update_vibration(self) -> None:
        cfg = self.cfg
        excitation = torch.randn(self.n, 6, generator=self.generator, device=self.device) * self.vibration_gain
        value = (self.vibration_phi * self.vibration_state).sum(1) + excitation
        self.vibration_state.copy_(torch.stack((value, self.vibration_state[:, 0]), 1))
        # 振动由自身指令运动激发（低通目标速度），接触时工具被约束不再叠加
        if cfg.target_lowpass:
            linear, angular = self.lowpass_v, self.lowpass_w
        else:
            _, q, v, angular = self.state()
            linear = v + torch.linalg.cross(angular, rotate(q, self.nominal_offset + self.calibration))
        speed = (torch.linalg.vector_norm(linear, dim=-1)
                 + cfg.vibration_moment_arm_m * torch.linalg.vector_norm(angular, dim=-1))
        free_air = (self.step_peak_contact <= CONTACT_THRESHOLD_N).float()
        motion = ((speed / cfg.vibration_speed_ref_mps).clamp(0., 1.) * free_air).unsqueeze(-1)
        scale = torch.cat((self.vibration_amplitude.expand(-1, 3),
                           self.vibration_amplitude.expand(-1, 3) * cfg.vibration_moment_arm_m), -1)
        self.vibration.copy_(value * scale * motion)

    def inject_step_bias(self) -> None:
        # 回合中途以泊松到达率注入一次持续的阶跃力旋量偏置，仅进入 actor 观测
        cfg = self.cfg
        probability = 1 - math.exp(-cfg.step_bias_rate_per_second * cfg.physics_dt * cfg.decimation)
        hit = (torch.rand(self.n, generator=self.generator, device=self.device) < probability) \
            & ~self.step_bias_active
        bias = torch.cat((
            self.unit_vector(self.n)*self.uniform(
                self.n, 1, cfg.step_force_bias_min_n, cfg.step_force_bias_max_n),
            self.unit_vector(self.n)*self.uniform(
                self.n, 1, cfg.step_moment_bias_min_nm, cfg.step_moment_bias_max_nm)), -1)
        self.wrench_bias.add_(bias * hit.unsqueeze(-1) * self.episode_noise_scale[:, None])
        self.step_bias_active |= hit

    def set_clearance(self, clearance: float) -> None:
        if self.adr:
            self.clearance_floor = clearance
            return
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
        everyone = torch.arange(self.n, device=self.device)
        self._init_shape()
        if self.cfg.shape_randomization:
            self.apply_shape(everyone)
        self.apply_geom_pos(everyone, grasp)
        if self.surf_slice is not None:
            self._randomize_rough_surface(everyone)
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
        self.masked_reset_graph = None
        self.masked_reset_candidate = None
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

    def warp_on_current_stream(self) -> wp.ScopedStream:
        # 重置中的 torch 与 warp 操作统一排在当前 torch 流上，保证顺序且可整体捕获
        return wp.ScopedStream(warp_stream(torch.cuda.current_stream(self.device)))

    def reset(self, ids: torch.Tensor) -> None:
        with self.warp_on_current_stream():
            self.reset_state(ids)
            self.finish_reset()

    def reset_masked(self, done: torch.Tensor) -> None:
        cfg = self.cfg
        if cfg.rough_surface or cfg.initial_plane_margin:
            # 初始位姿重采样循环依赖主机判断，只能按结束环境即时执行
            ids = done.nonzero().flatten()
            if len(ids):
                self.reset(ids)
            return
        self.reset_request.copy_(done)
        if not cfg.physics_cuda_graph:
            self.masked_reset()
            return
        signature = (self.difficulty, self.noise_scale, self.noise_floor, self.clearance,
                     self.clearance_floor, self.clearance_max)
        if self.masked_reset_graph is not None and signature == self.masked_reset_signature:
            self.masked_reset_graph.replay()
            return
        self.masked_reset()
        if signature == self.masked_reset_candidate:
            # 课程标量会固化进图，连续两次不变才捕获，按步变化的课程始终即时执行
            with torch.cuda.device(self.device):
                current = torch.cuda.current_stream(self.device)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(current)
                graph = torch.cuda.CUDAGraph()
                with wp.ScopedStream(wp.stream_from_torch(stream)):
                    with torch.cuda.graph(graph, stream=stream):
                        with wp.ScopedCapture(external=True, capture_mode=wp.CaptureMode.GLOBAL):
                            self.masked_reset(capturing=True)
                current.wait_stream(stream)
            self.masked_reset_graph, self.masked_reset_signature = graph, signature
        self.masked_reset_candidate = signature

    def per_env_state(self) -> list[tuple[torch.Tensor, int]]:
        """重置可能改写的逐环境张量及其环境维；广播常量（含 0 步长）不可写而跳过"""
        tensors = [t for t in vars(self).values() if isinstance(t, torch.Tensor)]
        tensors += [getattr(self.sim.model, name)[:] for name in self.expanded_fields]
        tensors += [getattr(self.sim.data, name)[:] for name in ("qpos", "qvel", "xfrc_applied")]
        state = []
        for t in tensors:
            if t.dim() == 0 or 0 in t.stride():
                continue
            if t.shape[0] == self.n:
                state.append((t, 0))
            elif t.dim() > 1 and t.shape[1] == self.n:
                state.append((t, 1))
        return state

    def masked_reset(self, capturing: bool = False) -> None:
        # 整组重新采样后只保留结束环境的新状态，张量形状固定可被图捕获
        mask = self.reset_request
        # 捕获时必须沿用开启捕获的同一 warp 流对象，否则 capture_while 误判为未捕获而同步
        with nullcontext() if capturing else self.warp_on_current_stream():
            saved = [(t, t.clone(), dim) for t, dim in self.per_env_state()]
            self.reset_state(self.everyone, mask)
            for t, old, dim in saved:
                keep = mask.view(*([1] * dim), -1, *([1] * (t.dim() - dim - 1)))
                t.copy_(torch.where(keep, t, old))
            self.finish_reset(capturing)

    def finish_reset(self, capturing: bool = False) -> None:
        cfg = self.cfg
        if capturing:
            self.sim.recompute_constants(RecomputeLevel.set_const)
        elif self.reset_constants_graph is not None and cfg.physics_cuda_graph:
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
        mjwarp.forward(self.sim.wp_model, self.sim.wp_data)

    def reset_state(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """采样并写入 ids 的新回合状态；物理常量与正向运动学由 finish_reset 统一重算"""
        n, cfg = len(ids), self.cfg
        if self.adr:
            u = self.uniform(n, 4, 0., 1.)
            probe = u[:, 0] < ADR_BOUNDARY_FRACTION
            clearance_probe = probe & (u[:, 1] < 0.5)
            noise_probe = probe & ~clearance_probe
            self.episode_boundary[ids] = noise_probe.long() + 2*clearance_probe.long()
            self.episode_noise_scale[ids] = self.noise_floor + torch.where(
                noise_probe, 1., u[:, 2]) * (self.noise_scale - self.noise_floor)
            low, high = math.log(self.clearance_floor), math.log(self.clearance_max)
            self.shape_params[ids, 3] = torch.where(
                clearance_probe, self.clearance_floor, (low + (high-low)*u[:, 3]).exp())
        else:
            self.episode_noise_scale.index_fill_(0, ids, self.noise_scale)
            self.shape_params[:, 3].index_fill_(0, ids, self.clearance)
        sim = self.sim
        if mask is None:
            sim._reset_mask.fill_(False)
            sim._reset_mask.index_fill_(0, ids, True)
        else:
            sim._reset_mask.copy_(mask)
        mjwarp.reset_data(sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp)
        low_depth, high_depth = cfg.target_depth_bounds_m
        if low_depth == high_depth:
            self.goal[:, 2].index_fill_(0, ids, -low_depth)
        else:
            self.goal[ids, 2] = -self.uniform(n, 1, low_depth, high_depth)[:, 0]
        if cfg.dynamics == "train":
            low, high, rot_low, rot_high = 120., 1200., 1.5, 10.
        elif cfg.dynamics == "soft":
            low, high, rot_low, rot_high = 120., 600., 1.5, 3.
        elif cfg.dynamics == "stiff":
            low, high, rot_low, rot_high = 3000., 5000., 15., 22.
        elif cfg.dynamics == "real":
            low = high = 650.
            rot_low = rot_high = 4.8
        else:
            low = high = 1500.
            rot_low = rot_high = 8.
        if cfg.dynamics == "train":
            self.k[ids] = torch.cat((
                self.uniform(n, 3, math.log(low), math.log(high)).exp(),
                self.uniform(n, 3, math.log(rot_low), math.log(rot_high)).exp()), -1)
        else:
            self.k[ids] = torch.cat((self.uniform(n, 3, low, high),
                                     self.uniform(n, 3, rot_low, rot_high)), -1)
        lower = self.constant(*[low]*3, *[rot_low]*3)
        upper = self.constant(*[high]*3, *[rot_high]*3)
        if cfg.training and cfg.dynamics == "train":
            center = self.constant(*[math.sqrt(low*high)]*3, *[math.sqrt(rot_low*rot_high)]*3)
            self.k[ids] = center * (self.k[ids] / center) ** self.difficulty
            lower = center * (lower / center) ** self.difficulty
            upper = center * (upper / center) ** self.difficulty
        self.initial_k[ids] = self.k[ids].double()
        self.walk_lower[ids] = torch.maximum(lower, self.initial_k[ids]*(1-cfg.impedance_walk_fraction))
        self.walk_upper[ids] = torch.minimum(upper, self.initial_k[ids]*(1+cfg.impedance_walk_fraction))
        self.walk_target[ids] = self.initial_k[ids]
        self.walk_current[ids] = self.initial_k[ids]
        # 轻刚体或低有效阻尼在 0.1 mm 间隙孔内会使 MuJoCo 接触发散，故刚体与阻尼按 0.7–1.5 kg；
        # 实机更轻的表观质量由目标低通加速度前馈 (m - m_apparent) * a_lowpass 与相对目标速度阻尼复现
        self.mass[ids] = self.uniform(n, 1, 0.7, 1.5)
        log_mass = (math.log(cfg.mass_min_kg), math.log(cfg.mass_max_kg))
        self.apparent_mass[ids] = self.curriculum_uniform(n, 1, *log_mass, sum(log_mass) / 2).exp()
        inertia_parts = self.uniform(n, 3, 0.001, 0.0025)
        self.inertia[ids] = (inertia_parts.sum(-1, keepdim=True)-inertia_parts) * cfg.scale**2
        if cfg.dynamics == "nominal":
            self.mass.index_fill_(0, ids, 1)
            self.apparent_mass.index_fill_(0, ids, 1)
            self.inertia.index_fill_(0, ids, 0.003 * cfg.scale**2)
        elif cfg.dynamics == "real":
            self.mass.index_fill_(0, ids, 1)
            self.apparent_mass.index_fill_(0, ids, 0.1)
            self.inertia.index_fill_(0, ids, 0.0024 * cfg.scale**2)
        if not cfg.target_lowpass:
            self.apparent_mass[ids] = self.mass[ids]
        inertia6 = torch.cat((self.apparent_mass[ids].expand(n, 3), self.inertia[ids]), -1)
        self.damping_ratio[ids] = self.curriculum_uniform(
            n, 6, cfg.damping_ratio_min, cfg.damping_ratio_max,
            min(max(1.0, cfg.damping_ratio_min), cfg.damping_ratio_max))
        if cfg.dynamics == "real":
            self.damping_ratio[ids] = self.constant(*[0.35]*3, *[0.5]*3)
        self.d[ids] = 2 * torch.sqrt(self.k[ids] * inertia6) * self.damping_ratio[ids]
        self.gain_q[ids] = rotvec_quat(self.uniform(n, 3, -0.5, 0.5))
        self.down_mass[ids] = self.curriculum_uniform(
            n, 1, cfg.down_mass_min_kg, cfg.down_mass_max_kg, (cfg.down_mass_min_kg + cfg.down_mass_max_kg) / 2)
        down_parts = self.uniform(n, 3, 0.00015, 0.00035)
        self.down_inertia[ids] = (down_parts.sum(-1, keepdim=True)-down_parts) * cfg.scale**2
        if cfg.dynamics == "nominal":
            self.damping_ratio.index_fill_(0, ids, 1)
            self.d[ids] = 2 * torch.sqrt(self.k[ids] * inertia6)
            self.gain_q[ids] = self.identity[ids]
            self.down_mass.index_fill_(0, ids, 0.3)
            self.down_inertia.index_fill_(0, ids, 0.0005 * cfg.scale**2)
        elif cfg.dynamics == "real":
            self.gain_q[ids] = self.identity[ids]
            self.down_mass.index_fill_(0, ids, 0.6)
        if cfg.sensor_tip_min_m is not None:
            fixed = cfg.dynamics in ("nominal", "real")
            mid = lambda low, high: torch.full((n, 1), (low + high) / 2, device=self.device)
            spread = lambda low, high: self.curriculum_uniform(n, 1, low, high, (low + high) / 2)
            d = (mid(cfg.sensor_tip_min_m, cfg.sensor_tip_max_m) if fixed else
                 spread(cfg.sensor_tip_min_m, cfg.sensor_tip_max_m)) * cfg.scale
            fraction = (mid(cfg.com_fraction_min, cfg.com_fraction_max) if fixed else
                        spread(cfg.com_fraction_min, cfg.com_fraction_max))
            lateral = (torch.zeros(n, 2, device=self.device) if fixed else
                       self.curriculum_uniform(n, 2, -cfg.com_lateral_m, cfg.com_lateral_m, 0.) * cfg.scale)
            sensor_z = self.nominal_offset[ids, 2:] + d
            self.sensor_offset[ids] = torch.cat((torch.zeros_like(lateral), sensor_z), -1)
            self.com_offset[ids] = torch.cat((lateral, sensor_z - fraction * d), -1)
            # 细杆近似：横向 m*d^2/12，轴向按 2–3.5 cm 半径
            transverse = self.down_mass[ids] * d**2 / 12 * (1.0 if fixed else self.uniform(n, 1, 0.6, 1.4))
            axial = self.down_mass[ids] * (2.5e-4 if fixed else
                                           self.uniform(n, 1, 1e-4, 4e-4)) * cfg.scale**2
            self.down_inertia[ids] = torch.cat((transverse, transverse, axial), -1)
        if cfg.target_lowpass:
            off = (torch.rand(n, 1, generator=self.generator, device=self.device)
                   < cfg.target_lowpass_off_probability)
            # 无低通时实机平移按表观质量的二阶响应追阶跃；上限保证 1 kHz 积分稳定且前馈不过度饱和
            natural = torch.sqrt(self.k[ids, :3] / self.apparent_mass[ids]).clamp_max(LOWPASS_OFF_MAX_RAD_S)
            self.lowpass_off[ids] = off
            self.lowpass_omega[ids] = torch.where(off, natural, cfg.target_lowpass_translation_rad_s)
            self.lowpass_zeta[ids] = torch.where(off, self.damping_ratio[ids, :3], 1.)
        stiction_force_z = cfg.stiction_force if cfg.stiction_force_z is None else cfg.stiction_force_z
        stiction_moment_xy = cfg.stiction_moment if cfg.stiction_moment_xy is None else cfg.stiction_moment_xy
        stiction_upper = [cfg.stiction_force]*2 + [stiction_force_z] + [stiction_moment_xy]*2 + [cfg.stiction_moment]
        if cfg.dynamics == "real":
            self.stiction[ids] = self.constant(0.3, 0.3, 2.5, 0.2, 0.2, 0.025)
        elif cfg.dynamics == "nominal" or not any(x > 0 for x in stiction_upper):
            self.stiction.index_fill_(0, ids, 0)
        else:
            self.stiction[ids] = self.uniform(n, 6, 0., 1.) * self.constant(*stiction_upper) * self.difficulty
            if cfg.stiction_reference_stiffness is not None:
                self.stiction[ids, :3] *= (self.k[ids, :3] / cfg.stiction_reference_stiffness).clamp_max(1.)
        if cfg.dynamics == "real":
            self.wall_friction.index_fill_(0, ids, 2.0)
        elif cfg.dynamics == "nominal" or cfg.wall_friction_force == 0:
            self.wall_friction.index_fill_(0, ids, 0)
        else:
            self.wall_friction[ids] = self.uniform(n, 1, 0., cfg.wall_friction_force)[:, 0] * self.difficulty
        # 实机绕尖端旋转时整臂绕上游枢轴转动，尖端随之平移
        self.pivot[ids] = self.unit_vector(n) * self.uniform(
            n, 1, 0., cfg.rotation_pivot_max_m * cfg.scale) * self.difficulty
        slip_angle = self.uniform(n, 1, -math.pi, math.pi)
        self.slip_amplitude[ids] = torch.cat(
            (slip_angle.cos(), slip_angle.sin(), torch.zeros_like(slip_angle)), -1) * self.uniform(
            n, 1, 0., cfg.stick_slip_max_pp_m * cfg.scale / 2) * self.difficulty
        self.slip_omega[ids] = 2 * math.pi * self.uniform(n, 1, cfg.stick_slip_min_hz, cfg.stick_slip_max_hz)[:, 0]
        self.slip_phase[ids] = self.uniform(n, 1, -math.pi, math.pi)[:, 0]
        if cfg.dynamics == "nominal":
            self.pivot.index_fill_(0, ids, 0)
            self.slip_amplitude.index_fill_(0, ids, 0)
        elif cfg.dynamics == "real":
            self.pivot[ids] = self.constant(0.05, 0., 0.05) * cfg.scale
            self.slip_amplitude[ids] = self.constant(0.0005, 0., 0.) * cfg.scale
            self.slip_omega.index_fill_(0, ids, 2 * math.pi * 5.0)
        self.clock.index_fill_(0, ids, 0)
        self.step_bias_active.index_fill_(0, ids, False)
        self.reset_vibration(ids)
        self.sim.model.body_mass[:][ids, self.body_id] = self.mass[ids, 0]
        self.sim.model.body_inertia[:][ids, self.body_id] = self.inertia[ids]
        friction = self.curriculum_uniform(n, 1, cfg.plate_friction_min, cfg.plate_friction_max,
                                           (cfg.plate_friction_min + cfg.plate_friction_max) / 2)
        if cfg.dynamics == "nominal":
            friction.fill_(0.4)
        elif cfg.dynamics == "real":
            friction.fill_(0.5)
        self.sim.model.geom_friction[:][ids, :, 0] = friction
        z_grasp_err = cfg.grasp_position_error if cfg.grasp_z_error is None else cfg.grasp_z_error
        if cfg.grasp_error_distribution == "gaussian":
            grasp_xy = self.normal(n, 2, cfg.grasp_position_error)
            grasp_z = (self.normal(n, 1, z_grasp_err).clamp(-2.5 * z_grasp_err, 2.5 * z_grasp_err)
                       if z_grasp_err > 0 else torch.zeros(n, 1, device=self.device))
            grasp = torch.cat((grasp_xy, grasp_z), dim=-1)
        else:
            grasp = torch.cat((
                self.uniform(n, 2, -cfg.grasp_position_error, cfg.grasp_position_error),
                self.uniform(n, 1, -z_grasp_err, z_grasp_err)), dim=-1)
        scale = self.episode_noise_scale[ids, None]
        grasp = grasp*scale
        self.offset[ids] = self.nominal_offset[ids] + grasp
        if cfg.shape_randomization:
            self.randomize_shape(ids)
        self.apply_geom_pos(ids, grasp)
        if cfg.rough_surface:
            self._randomize_rough_surface(ids)
        self.calibration[ids] = self.uniform(n, 3, -cfg.tip_calibration_error,
                                            cfg.tip_calibration_error)*scale
        self.bias[ids] = self.uniform(n, 3, -cfg.position_bias, cfg.position_bias)*scale
        if cfg.align_goal_z_to_observation:
            # 将名义目标深度同步平移纵向估计偏差 (h + Z)，使竖直到达实际深度 h 时 obs_z 恰好归零
            self.bias[ids, 2] = self.calibration[ids, 2] - grasp[:, 2]
        self.sensor_error[ids] = rotvec_quat(self.uniform(
            n, 3, -cfg.sensor_rotation_error, cfg.sensor_rotation_error)*scale)
        self.rotation_bias_q[ids] = rotvec_quat(self.uniform(
            n, 3, -cfg.rotation_bias, cfg.rotation_bias)*scale)
        if cfg.force_bias or cfg.moment_bias:
            self.wrench_bias[ids] = torch.cat((
                self.sample_wrench_bias(n, cfg.force_bias),
                self.sample_wrench_bias(n, cfg.moment_bias)), -1) * scale
        else:
            self.wrench_bias.index_fill_(0, ids, 0)
        self.delay[ids] = torch.randint(0, 3, (n,), generator=self.generator, device=self.device)
        if cfg.dynamics == "nominal":
            self.delay.index_fill_(0, ids, 0)
        elif cfg.dynamics == "real":
            self.delay.index_fill_(0, ids, 1)
        pos = self.uniform(n, 3, -cfg.initial_xy, cfg.initial_xy) * self.difficulty
        pos[:, 2] = self.curriculum_uniform(n, 1, cfg.initial_z_min, cfg.initial_z_max, cfg.initial_z_min)[:, 0]
        q = self.sample_initial_orientation(n)
        if cfg.initial_plane_margin or cfg.rough_surface:
            plane_margin = (max(cfg.initial_plane_margin, 0.0005 * cfg.scale)
                            if cfg.rough_surface else cfg.initial_plane_margin)
            surface_h = self.surface_max_height[ids] if cfg.rough_surface else None
            if (cfg.rough_surface and cfg.rough_low_spawn_prob > 0
                    and self.surf_tile_top_z is not None and self.surf_tile_xy is not None
                    and self.surf_horiz_radius is not None):
                poly_r = self.geometry.definition.peg_reach_radius(self.geometry, self.peg_polygon)
                peg_reach = poly_r + self.geometry.peg_length * math.sin(cfg.initial_tilt) + 0.001 * cfg.scale
                d_xy = torch.linalg.vector_norm(
                    pos[:, None, :2] - self.surf_tile_xy[ids], dim=-1)
                overlap = d_xy <= (peg_reach + self.surf_horiz_radius[None, :])
                masked_top = torch.where(
                    overlap, self.surf_tile_top_z[ids], torch.zeros_like(self.surf_tile_top_z[ids]))
                surface_h = masked_top.amax(dim=-1).clamp_min(0.0)
            for attempt in range(16):
                minimum_z = self.initial_minimum_tip_height(
                    q, margin=plane_margin, surface_height=surface_h, ids=ids)
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
            full_z = minimum_z + height_fraction * (cfg.initial_z_max-minimum_z)
            if cfg.rough_surface and cfg.rough_low_spawn_prob > 0:
                low_mask = self.uniform(n, 1, 0.0, 1.0)[:, 0] < cfg.rough_low_spawn_prob
                low_z = (minimum_z + height_fraction * (0.0025 * cfg.scale)).clamp_max(cfg.initial_z_max)
                pos[:, 2] = torch.where(low_mask, low_z, full_z)
            else:
                pos[:, 2] = full_z
        pos, q = self.sample_hard_start(ids, pos, q)
        qpos = self.sim.data.qpos[:]
        qpos[ids, :3] = pos - rotate(q, self.offset[ids])
        qpos[ids, 3:7] = q
        if cfg.rough_surface:
            self.tip_ema[ids] = torch.cat((pos[:, :2], pos[:, 2:3].clamp_max(0.0)), dim=-1)
        else:
            self.tip_ema[ids] = pos
        self.sim.data.qvel[:].index_fill_(0, ids, 0)
        self.sim.data.xfrc_applied[:].index_fill_(0, ids, 0)
        nominal_tip = tip_pose(qpos[ids, :3], q,
                               self.nominal_offset[ids] + self.calibration[ids])
        self.targets_p[:, ids] = nominal_tip.unsqueeze(0)
        self.targets_q[:, ids] = q.unsqueeze(0)
        self.lowpass_p[ids] = nominal_tip
        self.lowpass_q[ids] = q
        self.lowpass_v.index_fill_(0, ids, 0)
        self.lowpass_w.index_fill_(0, ids, 0)
        for buffer in (self.length, self.hold, self.peak_force, self.peak_moment,
                       self.step_peak_force, self.step_peak_moment,
                       self.last_action, self.wrench, self.raw_wrench, self.filtered_sensor,
                       self.hysteresis_sensor, self.wall_force_z, self.step_peak_contact):
            buffer.index_fill_(0, ids, 0)
        if cfg.position_observation_delay:
            self.reset_position_observation(ids, nominal_tip)

    def reset_position_observation(self, ids: torch.Tensor, observed_tip: torch.Tensor) -> None:
        n = len(ids)
        self.position_delay_frames[ids] = torch.randint(
            0, 2, (n,), generator=self.generator, device=self.device)
        position_noise = self.normal(n, 3, self.cfg.position_noise)*self.episode_noise_scale[ids, None]
        error = self.goal[ids] + self.bias[ids] - observed_tip - position_noise
        self.position_history[ids] = error[:, None, :].expand(-1, 2, -1)
        self.position_history_length[ids] = self.length[ids]

    def sample_initial_orientation(self, n: int) -> torch.Tensor:
        cfg = self.cfg
        rv = self.uniform(n, 3, -cfg.initial_tilt, cfg.initial_tilt) * self.difficulty
        rv[:, 2] = self.uniform(n, 1, -cfg.initial_yaw, cfg.initial_yaw)[:, 0] * self.difficulty
        return rotvec_quat(rv)

    def minimum_peg_height_offset(self, q: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        # 单位四元数旋转矩阵的第三行，即孔 Z 轴在插头坐标系中的方向
        w, x, y, z = q.unbind(dim=-1)
        local_vertical = torch.stack((2*(x*z-w*y), 2*(y*z+w*x),
                                      1-2*(x.square()+y.square())), dim=-1)
        params = self.shape_params if ids is None else self.shape_params[ids]
        radial_minimum = self.geometry.definition.minimum_peg_projection(
            self.geo, params, local_vertical[:, :2])
        return radial_minimum + (local_vertical[:, 2]*self.geometry.peg_length).clamp_max(0)

    def lowest_point_z(self, tip: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """插头底面最低点高度；孔口外的顶点（如落在围框外低处平面）不低于孔口平面"""
        definition = self.geometry.definition
        vertices = definition.peg_vertices(self.geo, self.shape_params)
        count = vertices.shape[1]
        local = torch.cat((vertices, torch.zeros_like(vertices[..., :1])), -1)
        world = tip[:, None] + rotate(q[:, None].expand(-1, count, -1), local)
        inside = definition.opening_contains(self.geo, self.shape_params, world[..., :2])
        z = world[..., 2]
        return torch.where(inside, z, z.clamp_min(0.)).amin(-1)

    def initial_minimum_tip_height(self, q: torch.Tensor, *,
                                   margin: float | None = None,
                                   surface_height: torch.Tensor | None = None,
                                   ids: torch.Tensor | None = None) -> torch.Tensor:
        lowest = self.minimum_peg_height_offset(q, ids)
        plane = self.cfg.initial_plane_margin if margin is None else margin
        base = plane - lowest if surface_height is None else surface_height + plane - lowest
        return base.clamp_min(self.cfg.initial_z_min)

    def phase(self) -> torch.Tensor:
        # 阶段监督标签（特权信息）：0 悬空，1 贴板但尖端不在开口投影内，2 尖端在开口投影内受阻，3 已入孔 0.5 mm 以上
        pos, q, _, _ = self.state()
        tip = tip_pose(pos, q, self.offset)
        over = (self.geometry.definition.opening_contains(self.geo, self.shape_params, tip[:, None, :2])[:, 0]
                if self.cfg.shape_randomization
                else torch.linalg.vector_norm(tip[:, :2], dim=-1) < self.geometry.radius)
        contact = self.step_peak_contact > CONTACT_THRESHOLD_N
        inserted = over & (tip[:, 2] < -0.0005)
        return torch.where(inserted, 3, torch.where(~contact, 0, torch.where(over, 2, 1)))

    def state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = self.sim.data.qpos[:, :3]
        q = canonical_quat(self.sim.data.qpos[:, 3:7])
        return pos, q, self.sim.data.qvel[:, :3], rotate(q, self.sim.data.qvel[:, 3:6])

    def impedance(self, error: torch.Tensor, velocity: torch.Tensor, part: slice,
                  feedforward: torch.Tensor | None = None,
                  goal_velocity: torch.Tensor | None = None) -> torch.Tensor:
        # 静摩擦死区作用于全部驱动项（弹簧、加速度前馈、目标速度阻尼），只有实际速度阻尼在死区外
        frame_inv = quat_inv(self.gain_q)
        local_e, local_v = rotate(frame_inv, error), rotate(frame_inv, velocity)
        spring = self.k[:, part] * local_e
        if feedforward is not None:
            spring = spring + rotate(frame_inv, feedforward)
        if goal_velocity is not None:
            spring = spring + self.d[:, part] * rotate(frame_inv, goal_velocity)
        spring = spring.sign() * (spring.abs() - self.stiction[:, part]).clamp_min(0.)
        return rotate(self.gain_q, spring - self.d[:, part] * local_v)

    def sample_impedance_walk_target(self) -> None:
        if self.cfg.dynamics in ("nominal", "real") or self.cfg.impedance_walk_fraction == 0:
            return
        interval = round(self.cfg.impedance_walk_interval_seconds / 0.02)
        due = (self.length % interval == 0).unsqueeze(-1)
        u = torch.rand(self.n, 6, generator=self.walk_generator, device=self.device, dtype=self.walk_target.dtype)
        self.walk_target.copy_(torch.where(
            due, self.walk_lower + u * (self.walk_upper - self.walk_lower), self.walk_target))

    def update_impedance_walk(self) -> None:
        cfg = self.cfg
        if cfg.dynamics in ("nominal", "real") or cfg.impedance_walk_fraction == 0:
            return
        # 使用 float64 累积物理子步，避免浮点舍入系统性突破变化率上限
        self.walk_current.copy_(bounded_stiffness_step(
            self.walk_current, self.walk_target, self.initial_k, self.walk_lower, self.walk_upper,
            cfg.physics_dt, cfg.impedance_walk_time_constant_seconds,
            cfg.impedance_walk_rate_per_second))
        self.k.copy_(self.walk_current)
        inertia6 = torch.cat((self.apparent_mass.expand(self.n, 3), self.inertia), -1)
        self.d.copy_(2 * self.damping_ratio * torch.sqrt(self.k * inertia6))

    def update_sensor(self) -> None:
        _, q, _, _ = self.state()
        acc = self.sim.data.qacc[:, :3]
        alpha_local = self.sim.data.qacc[:, 3:6]
        omega_local = self.sim.data.qvel[:, 3:6]
        contact_f = torch.cat((
            self.sim.data.qfrc_constraint[:, :2],
            (self.sim.data.qfrc_constraint[:, 2] + self.wall_force_z).unsqueeze(-1)), dim=-1)
        self.step_peak_contact.copy_(torch.maximum(
            self.step_peak_contact, torch.linalg.vector_norm(contact_f, dim=-1)))
        contact_m = rotate(q, self.sim.data.qfrc_constraint[:, 3:6])
        inertial_m = rotate(q, self.down_inertia * alpha_local
                            + torch.linalg.cross(omega_local, self.down_inertia * omega_local))
        r = self.com_offset
        acc_com = acc + rotate(q, torch.linalg.cross(alpha_local, r) + torch.linalg.cross(
            omega_local, torch.linalg.cross(omega_local, r)))
        inertial_f = self.down_mass * acc_com
        f = contact_f - inertial_f
        # 关于刚体原点的力矩；等效机械臂惯量不计入腕部传感器负载
        m_origin = contact_m - inertial_m - torch.linalg.cross(rotate(q, r), inertial_f)
        wrist = transform_wrench(f, m_origin, quat_inv(q), self.sensor_offset)
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
                            device=self.device) * noise_scale*self.episode_noise_scale[:, None]
        position_noise = self.normal(self.n, 3, self.cfg.position_noise)*self.episode_noise_scale[:, None]
        rotation_noise = self.normal(self.n, 3, self.cfg.rotation_noise)*self.episode_noise_scale[:, None]
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
        noisy_q = quat_mul(rotvec_quat(rotation_noise), quat_mul(self.rotation_bias_q, q))
        obs = actor_observation(noisy_tip, noisy_q, self.goal + self.bias,
                                self.identity, self.wrench + self.wrench_bias + self.vibration + noise)
        if self.cfg.position_observation_delay:
            delayed = self.position_history.gather(
                1, self.position_delay_frames[:, None, None].expand(-1, 1, 3)).squeeze(1)
            obs = torch.cat((delayed, obs[:, 3:]), dim=-1)
        clean = actor_observation(true_tip, q, self.goal, self.identity, self.raw_wrench)
        shape_id = torch.nn.functional.one_hot(
            torch.full((self.n,), ALL_SHAPES.index(self.shape), device=self.device),
            self.cfg.critic_shape_count).float()
        critic = torch.cat((clean, v, omega, self.k, self.d, self.apparent_mass, self.inertia,
                            self.down_mass, self.down_inertia, self.calibration,
                            quat_rotvec(self.sensor_error), self.bias,
                            self.offset-self.nominal_offset, shape_id), -1)
        if self.cfg.shape_randomization:
            critic = torch.cat((critic, self.shape_params*1000, self.rim_height[:, None]*1000), -1)
        if self.cfg.noise_curriculum_mode == "adr":
            # 同批混合不同噪声水平，力旋量干扰强度不可从其他特权量推断
            critic = torch.cat((critic, self.episode_noise_scale[:, None]), -1)
        return obs, critic

    def alignment(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(self.n, 3, device=self.device)
        z[:, 2] = 1
        axis = rotate(q, z)
        tilt = torch.atan2(torch.linalg.vector_norm(axis[:, :2], dim=-1), axis[:, 2])
        count = self.geometry.definition.rotation_order
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
        cfg = self.cfg
        goal_p, goal_q = self.physics_p_goal, self.physics_q_goal
        feedforward = None
        if cfg.target_lowpass:
            # 与部署侧一致的临界阻尼二阶目标低通，位于（延迟后的）目标与阻抗控制器之间
            dt = cfg.physics_dt
            omega_t, zeta_t = self.lowpass_omega, self.lowpass_zeta
            acceleration = (omega_t * omega_t * (goal_p - self.lowpass_p)
                            - 2 * zeta_t * omega_t * self.lowpass_v)
            feedforward = (self.mass - self.apparent_mass) * acceleration
            self.lowpass_v.add_(acceleration * dt)
            self.lowpass_p.add_(self.lowpass_v * dt)
            rotation_error = quat_rotvec(quat_mul(goal_q, quat_inv(self.lowpass_q)))
            self.lowpass_w.copy_(lowpass_velocity(
                self.lowpass_w, rotation_error, cfg.target_lowpass_rotation_rad_s, dt))
            self.lowpass_q.copy_(canonical_quat(quat_mul(rotvec_quat(self.lowpass_w * dt), self.lowpass_q)))
            goal_p = self.lowpass_p
            goal_q = torch.where(self.lowpass_off, goal_q, self.lowpass_q)
        self.clock.add_(cfg.physics_dt)
        stick_slip = self.slip_amplitude * torch.sin(self.slip_omega * self.clock + self.slip_phase).unsqueeze(-1)
        pos, q, v, omega = self.state()
        r_ct = rotate(q, self.nominal_offset + self.calibration)
        p_tip = pos + r_ct
        v_tip = v + torch.linalg.cross(omega, r_ct)
        goal_v = goal_w = None
        if feedforward is not None:
            goal_v = self.lowpass_v
            goal_w = torch.where(self.lowpass_off, 0., self.lowpass_w)
        force = self.impedance(goal_p + stick_slip - p_tip, v_tip, slice(0, 3),
                               feedforward, goal_v).clamp(-40, 40)
        angular_error = quat_rotvec(quat_mul(goal_q, quat_inv(q)))
        moment_tip = self.impedance(angular_error, omega, slice(3, 6), None, goal_w).clamp(-3, 3)
        true_tip = pos + rotate(q, self.offset)
        if self.cfg.shape_randomization:
            in_opening = self.geometry.definition.opening_contains(
                self.geo, self.shape_params, true_tip[:, None, :2])[:, 0].float()
        else:
            in_opening = (true_tip[:, :2].square().sum(-1) <= self.geometry.radius**2).float()
        depth_scale = ((-true_tip[:, 2] - 0.0005) / self.cfg.inserted_load_depth_m).clamp(0., 1.) * in_opening
        max_wall_f = self.wall_friction * depth_scale
        wall_f = (-force[:, 2] - self.mass[:, 0] * v[:, 2] / 0.005).clamp(-max_wall_f, max_wall_f)
        self.wall_force_z.copy_(wall_f)
        applied_f = torch.cat((force[:, :2], (force[:, 2] + wall_f).unsqueeze(-1)), dim=-1)
        self.applied_wrench[:, :3].copy_(applied_f)
        self.applied_wrench[:, 3:].copy_(moment_tip + torch.linalg.cross(r_ct, force))

    def physics_sensor(self) -> None:
        self.update_sensor()
        # 缓冲已满时约束会被静默丢弃，必须当作仿真失败处理
        self.step_contact_overflow.logical_or_(
            (self.nefc_view >= self.njmax) | (self.nacon_view >= self.naconmax))
        pos, q, _, _ = self.state()
        true_tip_z = (pos + rotate(q, self.offset))[:, 2]
        raw_hole_f = rotate(q, self.raw_wrench[:, :3])
        if self.cfg.inserted_load_cost:
            z_weight = (1.0 + true_tip_z / self.cfg.inserted_load_depth_m).clamp(0.0, 1.0)
            weighted_raw_f = torch.cat((
                raw_hole_f[:, :2], (raw_hole_f[:, 2] * z_weight).unsqueeze(-1)), dim=-1)
            step_force_norm = torch.linalg.vector_norm(weighted_raw_f, dim=-1)
        else:
            step_force_norm = torch.linalg.vector_norm(self.raw_wrench[:, :3], dim=-1)
        force_norm = torch.linalg.vector_norm(self.raw_wrench[:, :3], dim=-1)
        moment_norm = torch.linalg.vector_norm(self.raw_wrench[:, 3:], dim=-1)
        self.step_peak_force.copy_(torch.maximum(self.step_peak_force, step_force_norm))
        self.step_peak_moment.copy_(torch.maximum(self.step_peak_moment, moment_norm))
        self.peak_force.copy_(torch.maximum(self.peak_force, force_norm))
        self.peak_moment.copy_(torch.maximum(self.peak_moment, moment_norm))

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
        return self.finalize(*self.launch(actions))

    def launch(self, actions: torch.Tensor) -> tuple:
        # 只排队 GPU 工作、不做主机同步，使多个形状组可在不同 CUDA 流上并发
        if self.cfg.step_bias_rate_per_second > 0 and self.cfg.dynamics != "nominal":
            self.inject_step_bias()
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
        invalid = ~(torch.isfinite(self.sim.data.qpos[:]).all(-1) & torch.isfinite(self.sim.data.qvel[:]).all(-1)
                    & torch.isfinite(self.raw_wrench).all(-1)) | self.step_contact_overflow
        if self.cfg.vibration_force_max_n > 0 and self.cfg.dynamics != "nominal":
            self.update_vibration()
        if self.cfg.training and self.cfg.hard_state_reset_probability > 0:
            self.record_hard_states()
        reward, done, info = self.outcome(actions)
        return reward, done, info, invalid, self.last_reward_terms

    def finalize(self, reward: torch.Tensor, done: torch.Tensor, info: dict, invalid: torch.Tensor,
                 terms: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        self.last_reward_terms = terms
        failures = int(invalid.sum())
        if failures > 0.01 * self.n:
            raise FloatingPointError(f"invalid physical state in {self.shape}: {failures}/{self.n} environments")
        if failures:
            # 孔内接触偶发数值发散：仅终止该环境并按安全失败处罚，由上层正常重置
            logging.warning("MuJoCo divergence or contact overflow in %s: terminating %d environment(s) "
                            "as safety failures.",
                            self.shape, failures)
            reward = torch.where(invalid, -self.cfg.safety_failure_penalty, reward)
            done = done | invalid
            info = {key: torch.where(invalid, 0, value).to(value.dtype) if value.is_floating_point()
                    else value for key, value in info.items()}
            info["reason"] = torch.where(invalid, SIMULATION_FAILURE, info["reason"])
            info["time_outs"] = info["time_outs"] & ~invalid
            terms = {key: torch.where(invalid, 0., value) for key, value in self.last_reward_terms.items()}
            terms["safety_failure_penalty"] = torch.where(
                invalid, -self.cfg.safety_failure_penalty, terms["safety_failure_penalty"])
            self.last_reward_terms = terms
        info["simulation_failure"] = invalid
        return reward, done, info

    def outcome(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
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
        self.step_peak_force.zero_()
        self.step_peak_moment.zero_()
        self.step_peak_contact.zero_()
        self.step_contact_overflow.zero_()
        pos, q, _, _ = self.state()
        p_target, q_target = action_target(tip_pose(pos, q, self.nominal_offset + self.calibration),
                                           q, actions, cfg.translation_limit, cfg.rotation_limit)
        p_target = p_target + torch.linalg.cross(
            rotate(q, self.pivot), actions[:, 3:] * cfg.rotation_limit)
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
        lowest_z = (self.lowest_point_z(tip, q) if cfg.shape_randomization
                    else tip[:, 2] + self.minimum_peg_height_offset(q))
        z_disp = lowest_z - self.goal[:, 2]
        success = tip[:, 2] <= self.goal[:, 2] + 0.0002
        if cfg.success_opening_required:
            success = success & self.geometry.definition.inserted(self.geo, self.shape_params, tip, q)
        tip_center_z = tip[:, 2]
        if cfg.shape_randomization:
            # 落在围框外低处平面时不算入孔
            center_inside = self.geometry.definition.opening_contains(
                self.geo, self.shape_params, tip[:, None, :2])[:, 0]
            tip_center_z = torch.where(center_inside, tip_center_z, tip_center_z.clamp_min(0.))
        step_force = getattr(self, "step_peak_force", self.peak_force)
        step_moment = getattr(self, "step_peak_moment", self.peak_moment)
        overloaded = (step_force > cfg.max_force) | (step_moment > cfg.max_moment)
        if cfg.training and not cfg.terminate_on_overload:
            success = success & ~overloaded
        self.hold.copy_(torch.where(success, self.hold + 1, torch.zeros_like(self.hold)))
        success = self.hold >= cfg.success_hold_steps
        excessive = cfg.terminate_on_overload & (
            (self.peak_force > cfg.max_force) | (self.peak_moment > cfg.max_moment))
        escaped = ((xy > cfg.workspace_xy_radius_m) | (tip[:, 2] > cfg.workspace_z_max_m * cfg.scale)
                   | (tip[:, 2] < -(self.geometry.depth + (self.rim_height if cfg.shape_randomization else 0.)))
                   | (tilt > math.radians(35)))
        self.length += 1
        tip_ema = getattr(self, "tip_ema", None)
        stagnation = (None if tip_ema is None
                      else stagnation_level(cfg, tip, tip_ema, self.length, self.goal[:, 2]))
        timeout = ((self.length >= math.ceil(cfg.episode_seconds / 0.02))
                   if cfg.enable_timeout else torch.zeros_like(success))
        done = success | excessive | escaped | timeout
        success = success & ~excessive & ~escaped
        radius = self.opening_radius if cfg.shape_randomization else self.geometry.radius
        terms = completion_cost_terms(
            cfg, xy, z_disp, -self.goal[:, 2], radius, tilt, yaw,
            self.wrench, actions, self.last_action, success, excessive | escaped, timeout,
            tip_center_z=tip_center_z, hole_force=rotate(q, self.wrench[:, :3]),
            stagnation=stagnation,
            in_contact=(getattr(self, "step_peak_contact", step_force) > CONTACT_THRESHOLD_N))
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
                "time_outs": timeout & ~success & ~excessive & ~escaped,
                "hard_start": getattr(self, "hard_start", torch.zeros_like(success)).clone()}
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
        for d in dict.fromkeys((device, *devices)):
            if torch.device(d).type == "cuda":
                use_warp_stream_in_torch(d)
        self.groups = [
            PlantGroup(s, replace(cfg, shapes=(s,), shape_env_counts=(),
                                  envs_per_shape=count // len(devices),
                                  seed=cfg.seed + i*1000003), simulation_device)
            for s, count in zip(cfg.shapes, cfg.environment_counts, strict=True)
            for i, simulation_device in enumerate(devices)]
        self.num_envs = cfg.envs_per_shape * len(cfg.shapes)
        # 组流必须是 Warp 阻塞流：Warp 在空流上异步释放临时数组，只与阻塞流排序；
        # 非阻塞流上仍在执行的即时重置内核会读到已被其他流复用的内存，导致非法访存
        self.group_streams = [blocking_stream(g.device, i) for i, g in enumerate(self.groups)]
        self.warp_streams = [torch.cuda.ExternalStream(wp.get_stream(g.device).cuda_stream, device=g.device)
                             for g in self.groups]
        self.num_actions = 6
        self.max_episode_length = math.ceil(cfg.episode_seconds / 0.02)
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.policy_steps = 0
        self.total_transitions = 0
        use_success_curriculum = cfg.training and (
            cfg.noise_curriculum_mode in ("success", "adr") or cfg.clearance_curriculum)
        noise_levels = (10 if cfg.noise_curriculum and cfg.noise_curriculum_mode == "success"
                        else 0)
        self.noise_curriculum: SuccessNoiseCurriculum | ADRCurriculum | None = None
        if use_success_curriculum and cfg.noise_curriculum and cfg.noise_curriculum_mode == "adr":
            levels = cfg.active_clearance_levels
            self.noise_curriculum = ADRCurriculum(cfg.shapes, levels[0], levels[-1])
        elif use_success_curriculum:
            self.noise_curriculum = SuccessNoiseCurriculum(cfg.shapes, cfg.active_clearance_levels, noise_levels)
        self.curriculum_pending: list[torch.Tensor] = []
        self.curriculum_cohort: torch.Tensor | None = None
        self.curriculum_cohort_seen: torch.Tensor | None = None
        self.curriculum_cohort_armed: torch.Tensor | None = None
        self.curriculum_distributed = False
        self.curriculum_rollout_steps = 256
        self.curriculum_shape_ids = torch.cat([
            torch.full((g.n,), cfg.shapes.index(g.shape), device=device) for g in self.groups])

    def curriculum_noise_scale(self, group: PlantGroup) -> float:
        scale = None if self.noise_curriculum is None else self.noise_curriculum.scale_for(group.shape)
        return group.cfg.noise_scale(self.policy_steps) if scale is None else scale

    def apply_curriculum(self, group: PlantGroup) -> None:
        curriculum = self.noise_curriculum
        group.difficulty = curriculum.difficulty
        group.noise_scale = self.curriculum_noise_scale(group)
        if isinstance(curriculum, ADRCurriculum):
            group.noise_floor = curriculum.noise_floor_for(group.shape)
            group.clearance_max = curriculum.clearance_ceiling_for(group.shape)
        if hasattr(group, "set_clearance"):
            group.set_clearance(curriculum.clearance_for(group.shape))

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
            # 启动和恢复时物理环境都刚重置，首个队列从这些完整回合开始。
            self.curriculum_cohort_armed = torch.ones_like(self.curriculum_cohort_seen)
        cohort = self.curriculum_cohort
        seen = self.curriculum_cohort_seen
        armed = self.curriculum_cohort_armed
        assert seen is not None and armed is not None
        # 每个环境只计入队列开始后才开始的首个回合，避免长失败回合被按时长加权采样
        for batch in batches:
            done = batch[:, 1].bool()
            selected = done & batch[:, 3].bool() & armed & ~seen
            cohort[selected] = batch[selected]
            seen |= selected
            armed |= done
        complete = bool(seen.all())
        shapes = tuple(self.noise_curriculum.windows)
        recent = {}
        boundary = {}
        logs = {}
        for index, shape in enumerate(shapes):
            ended = (outcomes[:, 0] == index) & outcomes[:, 1].bool()
            completed = outcomes[ended, 2].bool()
            logs[f"success/{shape}"] = (completed.float().mean() if completed.numel()
                                        else outcomes.new_tensor(0., dtype=torch.float32))
            eligible = (cohort[:, 0] == index) & seen
            probe = cohort[:, 4].long()
            recent[shape] = cohort[eligible, 2].bool().cpu().tolist() if complete else []
            boundary[shape] = tuple(cohort[eligible & (probe == d), 2].bool().cpu().tolist() if complete else []
                                    for d in (1, 2))
        curriculum = self.noise_curriculum
        previous_scale = curriculum.scale
        previous_clearance = max(curriculum.clearance_for(s) for s in shapes)
        previous_difficulty = curriculum.difficulty
        upgraded = curriculum.update(recent, boundary) if complete else False
        if complete:
            seen.zero_()
            armed.zero_()
            for shape in shapes:
                for name, values in zip(("noise", "clearance"), boundary[shape]):
                    if values:
                        logs[f"curriculum/boundary_success/{name}/{shape}"] = outcomes.new_tensor(
                            sum(values)/len(values), dtype=torch.float32)
        adr = isinstance(curriculum, ADRCurriculum)
        if upgraded and adr:
            logging.info("ADR: difficulty %.2f -> %.2f, phi (noise, clearance) %s",
                         previous_difficulty, curriculum.difficulty, curriculum.phi)
        elif upgraded:
            logging.info(
                "Success curriculum level %d: clearance %.2f -> %.2f mm, "
                "difficulty %.2f -> %.2f, noise %.1f -> %.1f; "
                "all shapes reached 80%% over 1000 episodes.",
                self.noise_curriculum.level, previous_clearance * 1000,
                self.noise_curriculum.clearance * 1000, previous_difficulty,
                self.noise_curriculum.difficulty, previous_scale, self.noise_curriculum.scale,
            )
        for group in self.groups:
            self.apply_curriculum(group)
        logs["curriculum/clearance_mm"] = outcomes.new_tensor(
            max(curriculum.clearance_for(s) for s in shapes) * 1000, dtype=torch.float32)
        logs["curriculum/difficulty"] = outcomes.new_tensor(
            self.noise_curriculum.difficulty, dtype=torch.float32)
        logs["curriculum/noise_scale"] = outcomes.new_tensor(
            self.noise_curriculum.scale, dtype=torch.float32)
        if adr:
            for shape, (noise_phi, clearance_phi) in curriculum.phi.items():
                logs[f"curriculum/noise_phi/{shape}"] = outcomes.new_tensor(noise_phi, dtype=torch.float32)
                logs[f"curriculum/clearance_phi/{shape}"] = outcomes.new_tensor(
                    clearance_phi, dtype=torch.float32)
                logs[f"curriculum/clearance_mm/{shape}"] = outcomes.new_tensor(
                    curriculum.clearance_for(shape) * 1000, dtype=torch.float32)
        for shape, window in self.noise_curriculum.windows.items():
            logs[f"curriculum/episodes/{shape}"] = outcomes.new_tensor(len(window))
            logs[f"curriculum/success/{shape}"] = outcomes.new_tensor(
                sum(window)/len(window) if window else 0., dtype=torch.float32)
        return logs

    def launch_groups(self, chunks: list[torch.Tensor]) -> list[tuple]:
        return self.concurrent_groups(lambda g, a: g.launch(a), chunks)

    def concurrent_groups(self, work, inputs: list) -> list:
        # 仿真受内核串行延迟而非算力限制，各形状组在独立流上并发；仅在整步物理已被 torch 图捕获时启用
        if not all(g.physics_rollout_graph is not None for g in self.groups):
            return [work(g, x) for g, x in zip(self.groups, inputs, strict=True)]
        results = []
        for g, x, stream, warp_stream in zip(self.groups, inputs, self.group_streams, self.warp_streams,
                                             strict=True):
            stream.wait_stream(torch.cuda.current_stream(g.device))
            stream.wait_stream(warp_stream)
            with torch.cuda.stream(stream):
                results.append(work(g, x))
        for g, stream, warp_stream in zip(self.groups, self.group_streams, self.warp_streams, strict=True):
            torch.cuda.current_stream(g.device).wait_stream(stream)
            warp_stream.wait_stream(stream)
        return results

    def get_observations(self) -> TensorDict:
        pairs = [g.observations() for g in self.groups]
        # free_air 不进入任何网络，仅供 SmoothPPO 选择悬空步
        free_air = torch.cat([(g.step_peak_contact <= CONTACT_THRESHOLD_N).float().to(self.device)
                              for g in self.groups]).unsqueeze(-1)
        # noise_scale 不进入任何网络，供 SmoothPPO 按当前噪声课程缩放悬空扰动
        noise_scale = torch.cat([g.episode_noise_scale.to(self.device) for g in self.groups]).unsqueeze(-1)
        # phase 不进入任何网络，仅供 SmoothPPO 的阶段辅助头作监督标签
        phase = torch.cat([g.phase().to(self.device) for g in self.groups]).unsqueeze(-1)
        return TensorDict({"actor": torch.cat([p[0].to(self.device) for p in pairs]),
                           "critic": torch.cat([p[1].to(self.device) for p in pairs]),
                           "free_air": free_air, "noise_scale": noise_scale, "phase": phase},
                          [self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, 6):
            raise ValueError(f"expected {(self.num_envs, 6)} action, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite policy action; no silent reset")
        chunks = [a.to(g.device) for g, a in zip(
            self.groups, actions.split([g.n for g in self.groups]), strict=True)]
        results = [g.finalize(*p) for g, p in zip(self.groups, self.launch_groups(chunks), strict=True)]
        self.policy_steps += 1
        self.total_transitions += self.num_envs
        for g in self.groups:
            g.noise_scale = self.curriculum_noise_scale(g)
            if g.cfg.training:
                g.difficulty = (self.noise_curriculum.difficulty
                                if self.noise_curriculum is not None
                                else min(1., self.policy_steps / g.cfg.curriculum_steps))
        rewards = torch.cat([r[0].to(self.device) for r in results])
        dones = torch.cat([r[1].to(self.device) for r in results])
        terminal = {key: torch.cat([r[2][key].to(self.device) for r in results])
                    for key in results[0][2]}
        if self.noise_curriculum is not None:
            episode_scales = torch.cat([g.episode_noise_scale.to(self.device) for g in self.groups])
            episode_clearances = torch.cat([g.shape_params[:, 3].to(self.device) for g in self.groups])
            probes = torch.cat([g.episode_boundary.to(self.device) for g in self.groups])
            scale_target = torch.cat([
                torch.full((g.n,), s, device=self.device)
                if (s := self.noise_curriculum.scale_for(g.shape)) is not None
                else g.episode_noise_scale.to(self.device) for g in self.groups])
            clearance_target = torch.cat([
                torch.full((g.n,), self.noise_curriculum.clearance_for(g.shape), device=self.device)
                for g in self.groups])
            scale_ok = torch.isclose(episode_scales, scale_target)
            clearance_ok = torch.isclose(episode_clearances, clearance_target, rtol=0., atol=1e-9)
            if isinstance(self.noise_curriculum, ADRCurriculum):
                # 普通回合两维本就随机；边界回合须钉在当前边界（课程中途变化或困难状态回放的不计）
                eligible = (probes == 0) | ((probes == 1) & scale_ok) | ((probes == 2) & clearance_ok)
            else:
                eligible = scale_ok & clearance_ok
            terminal["curriculum_boundary"] = probes
            terminal["curriculum_eligible"] = eligible
            if not self.curriculum_distributed:
                self.record_curriculum_outcomes(torch.stack((
                    self.curriculum_shape_ids, dones, terminal["reason"] == 1,
                    terminal["curriculum_eligible"], probes), dim=1))
        logs = {}
        logs["episode/simulation_failures"] = terminal["simulation_failure"].float().sum()
        hard = terminal["hard_start"] & dones
        logs["episode/hard_start_fraction"] = hard.float().sum() / dones.float().sum().clamp_min(1)
        logs["episode/hard_start_success"] = (
            (hard & (terminal["reason"] == 1)).float().sum() / hard.float().sum().clamp_min(1))
        logs["curriculum/noise_scale"] = torch.tensor(
            self.groups[0].noise_scale, device=self.device)
        for key in self.groups[0].last_reward_terms:
            logs[f"reward/{key}"] = torch.stack(
                [g.last_reward_terms[key].mean().to(self.device)
                 for g in self.groups]).mean()
        successes = {}
        for group, (_, done, info) in zip(self.groups, results, strict=True):
            if self.noise_curriculum is None and bool(done.any()):
                successes.setdefault(group.shape, []).append(
                    torch.stack((((info["reason"] == 1) & done).float().sum(),
                                 done.float().sum())).to(self.device))
        # 一次同步取得各组是否有结束环境，无结束的组跳过重置且不消耗随机数
        dones_by_group = [r[1] for r in results]
        active = torch.stack([d.any().to(self.device) for d in dones_by_group]).tolist()
        self.concurrent_groups(lambda g, x: g.reset_masked(x[0]) if x[1] else None,
                               list(zip(dones_by_group, active)))
        for shape, counts in successes.items():
            totals = torch.stack(counts).sum(dim=0)
            if self.noise_curriculum is None:
                logs[f"success/{shape}"] = totals[0]/totals[1]
        self.episode_length_buf.copy_(torch.cat([g.length.to(self.device) for g in self.groups]))
        if not self.groups[0].cfg.enable_timeout or self.groups[0].cfg.rough_surface:
            logs["episode/active_max_steps"] = self.episode_length_buf.max().float()
            logs["episode/stuck_over_8s_fraction"] = (self.episode_length_buf > 400).float().mean()
            logs["episode/stuck_over_30s_fraction"] = (self.episode_length_buf > 1500).float().mean()
        if self.noise_curriculum is not None:
            logs.pop("curriculum/noise_scale")
            if not self.curriculum_distributed and self.policy_steps % self.curriculum_rollout_steps == 0:
                logs.update(self.advance_noise_curriculum())
        # 超时视为真实任务失败，不向 PPO 发送超时价值自举信号
        return self.get_observations(), rewards, dones, {
            "time_outs": torch.zeros_like(terminal["time_outs"]),
            "log": logs, "terminal": terminal}
