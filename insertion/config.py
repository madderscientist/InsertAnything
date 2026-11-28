"""仿真与训练参数配置，仅用于物理仿真环境"""

from dataclasses import asdict, dataclass, fields
import math

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .shapes import ALL_SHAPES, LEGACY_SHAPE_COUNT, TRAIN_SHAPES
from .shapes.circle import validate_circle_sides

HOLE_DEPTH_M = 0.025
PPO_DISCOUNT = 0.9985
CLEARANCE_CURRICULUM_LEVELS = (0.002, 0.001, 0.0005, 0.00025, 0.00015, 0.0001)
# real2sim/ 续训覆盖到源配置上的真机实测取值；数据类默认值为第一阶段（无部署低通）
REAL2SIM_VALUES = {
    "initial_z_max": 0.055, "workspace_z_max_m": 0.070,
    "action_change_cost_per_second": 0.2, "free_air_action_change_cost_per_second": 0.6,
    "action_cost_cap": 12.0,
    "stiction_force_z": 3.0, "stiction_moment_xy": 0.25, "stiction_reference_stiffness": 650.0,
    "mass_min_kg": 0.08, "damping_ratio_min": 0.25, "plate_friction_min": 0.15,
    "down_mass_min_kg": 0.4, "down_mass_max_kg": 0.9,
    "sensor_tip_min_m": 0.18, "sensor_tip_max_m": 0.22,
    "com_fraction_min": 0.3, "com_fraction_max": 0.6, "com_lateral_m": 0.005,
    "target_lowpass": True, "target_lowpass_translation_rad_s": 30.0,
    "target_lowpass_rotation_rad_s": 50.0, "target_lowpass_off_probability": 1.0,
    "rotation_pivot_max_m": 0.09,
    "stick_slip_max_pp_m": 0.0015, "stick_slip_min_hz": 3.0, "stick_slip_max_hz": 6.0,
    "step_bias_rate_per_second": 0.05, "step_force_bias_min_n": 1.0, "step_force_bias_max_n": 3.5,
    "step_moment_bias_min_nm": 0.1, "step_moment_bias_max_nm": 0.6,
    "vibration_force_min_n": 0.5, "vibration_force_max_n": 1.5,
    "vibration_min_hz": 2.0, "vibration_max_hz": 10.0,
    "vibration_moment_arm_m": 0.1, "vibration_speed_ref_mps": 0.02,
    "hard_state_reset_probability": 0.25,
}
REAL2SIM_FIELDS = tuple(REAL2SIM_VALUES)


@dataclass
class InsertionCfg:
    shapes: tuple[str, ...] = TRAIN_SHAPES
    envs_per_shape: int = 16
    shape_env_counts: tuple[int, ...] = ()
    adaptive_sampling: bool = False
    seed: int = 42
    physics_dt: float = 0.001
    decimation: int = 20
    physics_cuda_graph: bool = False
    observation_cuda_graph: bool = False
    position_observation_delay: bool = True
    inserted_load_cost: bool = True
    inserted_load_depth_m: float = 0.003
    episode_seconds: float = 8.0
    enable_timeout: bool = True
    rough_surface: bool = False
    rough_min_amplitude_m: float = 0.0
    rough_max_amplitude_m: float = 0.0025
    rough_wavelength_min_m: float = 0.008
    rough_wavelength_max_m: float = 0.036
    rough_max_slope_deg: float = 18.0
    rough_flat_probability: float = 0.1
    rough_num_waves: int = 4
    rough_step_jitter: float = 0.0
    rough_stepped_prob: float = 0.0
    rough_low_spawn_prob: float = 0.0
    clearance: float = 0.0001
    circle_sides: int | None = None
    clearance_curriculum: bool = False
    clearance_levels: tuple[float, ...] = CLEARANCE_CURRICULUM_LEVELS
    scale: float = 1.0
    hole_depth_m: float = HOLE_DEPTH_M
    target_depth_min_m: float = 0.005
    target_depth_max_fraction: float = 0.9
    fixed_target_depth_m: float | None = None
    align_goal_z_to_observation: bool = True
    translation_limit: float = 0.001
    rotation_limit: float = math.radians(0.5)
    max_force: float = 30.0
    max_moment: float = 2.0
    terminate_on_overload: bool = False
    filter_seconds: float = 0.015
    wrench_hysteresis_fraction: float = 0.001
    initial_xy: float = 0.020
    workspace_xy_radius_m: float = 0.035
    workspace_z_max_m: float = 0.070
    plate_outer_radius_m: float = 0.080
    initial_z_min: float = 0.004
    initial_z_max: float = 0.055
    initial_plane_margin: float = 0.0
    initial_tilt: float = math.radians(8)
    initial_yaw: float = math.radians(15)
    position_bias: float = 0.0005
    tip_calibration_error: float = 0.0005
    grasp_position_error: float = 0.0015
    grasp_z_error: float | None = None
    grasp_error_distribution: str = "gaussian"
    sensor_rotation_error: float = math.radians(0.5)
    rotation_bias: float = math.radians(1.5)
    position_noise: float = 0.0005
    rotation_noise: float = math.radians(1.5)
    force_noise: float = 0.2
    moment_noise: float = 0.015
    force_bias: float = 0.5
    moment_bias: float = 0.1
    wrench_bias_distribution: str = "uniform_magnitude"
    noise_curriculum: bool = False
    noise_curriculum_mode: str = "steps"
    noise_zero_steps: int = 25600
    noise_full_steps: int = 204800
    dynamics: str = "train"
    training: bool = True
    success_hold_steps: int = 5
    success_opening_required: bool = True
    curriculum_steps: int = 256000
    impedance_walk_fraction: float = 0.1
    impedance_walk_rate_per_second: float = 0.05
    impedance_walk_interval_seconds: float = 1.0
    impedance_walk_time_constant_seconds: float = 1.0
    stiction_force: float = 0.6
    stiction_moment: float = 0.06
    # None 表示与 stiction_force/stiction_moment 相同；real2sim 使用实测 Z/XY 转动独立死区
    stiction_force_z: float | None = None
    stiction_moment_xy: float | None = None
    # 设定时低于该刚度按比例缩小平移死区，保持等效位移不超过标定值
    stiction_reference_stiffness: float | None = None
    # 阻抗表观质量（仅启用目标低通时生效）；MuJoCo 刚体质量固定在 0.7–1.5 kg
    mass_min_kg: float = 0.7
    mass_max_kg: float = 1.5
    damping_ratio_min: float = 0.25
    damping_ratio_max: float = 1.3
    down_mass_min_kg: float = 0.4
    down_mass_max_kg: float = 0.9
    # 传感器沿轴位于尖端上方 d；下游质心在传感器下方 fraction*d 并有横向偏移；None 为旧模型
    sensor_tip_min_m: float | None = 0.08
    sensor_tip_max_m: float = 0.22
    com_fraction_min: float = 0.3
    com_fraction_max: float = 0.6
    com_lateral_m: float = 0.005
    plate_friction_min: float = 0.25
    plate_friction_max: float = 1.15
    target_lowpass: bool = False
    target_lowpass_translation_rad_s: float = 30.0
    target_lowpass_rotation_rad_s: float = 50.0
    # 以此概率按回合关闭部署低通：平移改为轻表观质量阻抗的自身响应 sqrt(k/m)、转动直接跟随阶跃目标
    target_lowpass_off_probability: float = 0.0
    rotation_pivot_max_m: float = 0.0
    stick_slip_max_pp_m: float = 0.0
    stick_slip_min_hz: float = 3.0
    stick_slip_max_hz: float = 6.0
    step_bias_rate_per_second: float = 0.0
    step_force_bias_min_n: float = 1.0
    step_force_bias_max_n: float = 3.5
    step_moment_bias_min_nm: float = 0.1
    step_moment_bias_max_nm: float = 0.6
    # 训练时以此概率从近期记录的接触难点位姿（贴板、孔口卡滞、半插入）开始回合
    hard_state_reset_probability: float = 0.25
    # 逐环境随机形状尺寸，并以一定概率在孔口加一圈高出周围平面的围框（孔口顶面始终为 z=0）
    shape_randomization: bool = False
    rim_probability: float = 0.5
    rim_height_min_m: float = 0.0002
    rim_height_max_m: float = 0.002
    rim_width_min_m: float = 0.0025
    rim_width_max_m: float = 0.0035
    # 运动引起的夹爪/工具振动力：2–10 Hz 低阻尼窄带有色噪声，幅值随尖端速度增长，仅进入 actor 观测
    vibration_force_min_n: float = 0.5
    vibration_force_max_n: float = 1.5
    vibration_min_hz: float = 2.0
    vibration_max_hz: float = 10.0
    vibration_moment_arm_m: float = 0.1
    vibration_speed_ref_mps: float = 0.02
    wall_friction_force: float = 4.0
    time_cost_per_second: float = 1.0
    position_cost_per_second: float = 2.0
    position_cost_cap: float = 2.0
    surface_z_free_m: float = 0.0
    xy_cost_radius_fraction: float = 0.25
    xy_outer_cost_weight: float = 0.2
    z_outer_cost_weight: float = 0.2
    orientation_cost_per_second: float = 1.0
    orientation_cost_reference_deg: float = 5.0
    force_cost_per_second: float = 1.0
    moment_cost_per_second: float = 1.0
    load_cost_cap: float = 1.5
    stagnation_cost_per_second: float = 2.0
    stagnation_speed_threshold_mps: float = 0.006
    stagnation_window_seconds: float = 0.4
    force_cost_free_n: float = 5.0
    moment_cost_free_nm: float = 0.1
    action_cost_per_second: float = 0.05
    action_change_cost_per_second: float = 0.1
    # 悬空时更强的动作变化成本；接触时允许晃动以搜索孔口
    free_air_action_change_cost_per_second: float | None = 0.5
    free_air_action_change_squared: bool = False
    # 让 ~20 单位的大动作仍有代价梯度
    action_cost_cap: float = 12.0
    completion_bonus: float = 20.0
    safety_failure_penalty: float = 250.0
    timeout_penalty: float = 10.0
    # 入孔深度整形 Φ = w·入孔深度/目标深度，奖励加 γΦ' − Φ；终止态保留 Φ'，即超时按终止深度给部分分
    # （终止 Φ'=0 的纯势函数整形会被 critic 吸收，优势与不整形时相同）
    insertion_depth_reward: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.terminate_on_overload, bool):
            raise ValueError("terminate_on_overload must be a boolean")
        if not isinstance(self.enable_timeout, bool) or not isinstance(self.rough_surface, bool):
            raise ValueError("enable_timeout and rough_surface must be booleans")
        if not isinstance(self.align_goal_z_to_observation, bool):
            raise ValueError("align_goal_z_to_observation must be a boolean")
        if not math.isfinite(self.hole_depth_m) or not 0.01 <= self.hole_depth_m <= 0.033:
            raise ValueError("hole_depth_m must be finite and within [0.01, 0.033] m (below 35 mm peg length)")
        if self.grasp_z_error is not None and (not math.isfinite(self.grasp_z_error) or self.grasp_z_error < 0):
            raise ValueError("grasp_z_error must be None or finite and nonnegative")
        if not math.isfinite(self.rough_max_amplitude_m) or self.rough_max_amplitude_m < 0:
            raise ValueError("rough_max_amplitude_m must be finite and nonnegative")
        if (not math.isfinite(self.rough_wavelength_min_m)
                or not math.isfinite(self.rough_wavelength_max_m)
                or not 0 < self.rough_wavelength_min_m <= self.rough_wavelength_max_m):
            raise ValueError("rough surface wavelengths must be positive and ordered")
        if not math.isfinite(self.rough_max_slope_deg) or not 0 < self.rough_max_slope_deg <= 45:
            raise ValueError("rough_max_slope_deg must be finite and in (0, 45]")
        if not math.isfinite(self.rough_flat_probability) or not 0 <= self.rough_flat_probability <= 1:
            raise ValueError("rough_flat_probability must be finite and in [0, 1]")
        if type(self.rough_num_waves) is not int or not 1 <= self.rough_num_waves <= 16:
            raise ValueError("rough_num_waves must be an integer in [1, 16]")
        if not math.isfinite(self.inserted_load_depth_m) or self.inserted_load_depth_m <= 0:
            raise ValueError("inserted_load_depth_m must be finite and positive")
        validate_circle_sides(self.circle_sides)
        self.shapes = tuple(self.shapes)
        if not self.shapes or any(s not in ALL_SHAPES for s in self.shapes):
            raise ValueError(f"shapes must belong to {ALL_SHAPES}")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("duplicate geometry groups are not allowed")
        if self.envs_per_shape < 1 or self.decimation < 1:
            raise ValueError("environment count and decimation must be positive")
        self.shape_env_counts = tuple(self.shape_env_counts)
        if self.shape_env_counts and (
                len(self.shape_env_counts) != len(self.shapes)
                or any(type(n) is not int or n < 1 for n in self.shape_env_counts)
                or sum(self.shape_env_counts) != self.envs_per_shape * len(self.shapes)):
            raise ValueError("shape_env_counts must be positive integers preserving the total environment count")
        if self.adaptive_sampling and (not self.training or not (
                self.clearance_curriculum or self.noise_curriculum_mode in ("success", "adr"))):
            raise ValueError("adaptive sampling requires training with a success-based curriculum")
        if not math.isclose(self.physics_dt * self.decimation, 0.02, abs_tol=1e-10):
            raise ValueError("policy period must be 0.02 s (50 Hz)")
        positive = (self.clearance, self.scale, self.episode_seconds,
                    self.translation_limit, self.rotation_limit, self.max_force,
                    self.max_moment, self.filter_seconds)
        if any(x <= 0 or not math.isfinite(x) for x in positive):
            raise ValueError("physical scales, periods and limits must be finite and positive")
        self.clearance_levels = tuple(float(c) for c in self.clearance_levels)
        max_clearance = 0.005 * self.scale
        if (not self.clearance_levels
                or any(not math.isfinite(c) or not 0 < c < max_clearance
                       for c in self.clearance_levels)
                or any(a <= b for a, b in zip(self.clearance_levels, self.clearance_levels[1:]))):
            raise ValueError("clearance_levels must be finite, strictly decreasing and < 0.005 * scale")
        if self.clearance_curriculum and self.clearance_levels[0] <= self.clearance:
            raise ValueError("clearance curriculum requires an initial clearance above target clearance")
        if self.dynamics not in ("train", "nominal", "soft", "stiff", "real"):
            raise ValueError("unknown dynamics profile")
        if not isinstance(self.target_lowpass, bool):
            raise ValueError("target_lowpass must be a boolean")
        if not 0 <= self.target_lowpass_off_probability <= 1:
            raise ValueError("target_lowpass_off_probability must be in [0, 1]")
        ranges = ((self.mass_min_kg, self.mass_max_kg), (self.damping_ratio_min, self.damping_ratio_max),
                  (self.down_mass_min_kg, self.down_mass_max_kg),
                  (self.plate_friction_min, self.plate_friction_max),
                  (self.stick_slip_min_hz, self.stick_slip_max_hz),
                  (self.step_force_bias_min_n, self.step_force_bias_max_n),
                  (self.step_moment_bias_min_nm, self.step_moment_bias_max_nm),
                  (self.vibration_min_hz, self.vibration_max_hz))
        if any(not (math.isfinite(a) and math.isfinite(b) and 0 < a <= b) for a, b in ranges):
            raise ValueError("plant randomization ranges must be finite, positive and ordered")
        if self.sensor_tip_min_m is not None and (
                not 0 < self.sensor_tip_min_m <= self.sensor_tip_max_m < 1
                or not 0 <= self.com_fraction_min <= self.com_fraction_max <= 1
                or not 0 <= self.com_lateral_m < 0.05):
            raise ValueError("sensor/COM geometry ranges must be ordered and physical")
        if (not math.isfinite(self.vibration_force_max_n)
                or not 0 <= self.vibration_force_min_n <= self.vibration_force_max_n
                or not math.isfinite(self.vibration_moment_arm_m) or self.vibration_moment_arm_m < 0
                or not math.isfinite(self.vibration_speed_ref_mps) or self.vibration_speed_ref_mps <= 0
                or self.vibration_max_hz >= 0.5 / (self.physics_dt * self.decimation)):
            raise ValueError("vibration noise must be finite, ordered and below the policy Nyquist rate")
        if not math.isfinite(self.hard_state_reset_probability) or not 0 <= self.hard_state_reset_probability <= 1:
            raise ValueError("hard_state_reset_probability must be finite and in [0, 1]")
        if self.stiction_reference_stiffness is not None and (
                not math.isfinite(self.stiction_reference_stiffness) or self.stiction_reference_stiffness <= 0):
            raise ValueError("stiction reference stiffness must be None or finite and positive")
        if self.free_air_action_change_cost_per_second is not None and (
                not math.isfinite(self.free_air_action_change_cost_per_second)
                or self.free_air_action_change_cost_per_second < 0):
            raise ValueError("free-air action change cost must be None or finite and nonnegative")
        plant_nonnegative = (self.rotation_pivot_max_m, self.stick_slip_max_pp_m,
                             self.step_bias_rate_per_second,
                             *(x for x in (self.stiction_force_z, self.stiction_moment_xy) if x is not None))
        if any(not math.isfinite(x) or x < 0 for x in plant_nonnegative):
            raise ValueError("plant disturbance magnitudes must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in (
                self.target_lowpass_translation_rad_s, self.target_lowpass_rotation_rad_s)):
            raise ValueError("target low-pass natural frequencies must be finite and positive")
        if self.success_hold_steps < 1 or self.curriculum_steps < 1:
            raise ValueError("success hold and curriculum horizon must be positive")
        noise = (self.initial_xy, self.initial_tilt, self.initial_yaw, self.position_bias,
                 self.tip_calibration_error, self.grasp_position_error, self.sensor_rotation_error,
                 self.rotation_bias,
                 self.position_noise, self.rotation_noise, self.force_noise, self.moment_noise,
                 self.force_bias, self.moment_bias, self.wrench_hysteresis_fraction,
                 self.stiction_force, self.stiction_moment, self.wall_friction_force)
        if any(x < 0 or not math.isfinite(x) for x in noise):
            raise ValueError("randomization magnitudes must be finite and nonnegative")
        if self.wrench_bias_distribution not in ("uniform_magnitude", "per_axis"):
            raise ValueError("wrench bias distribution must be uniform_magnitude or per_axis")
        if (not math.isfinite(self.workspace_xy_radius_m) or self.workspace_xy_radius_m <= 0
                or math.sqrt(2)*self.initial_xy > self.workspace_xy_radius_m):
            raise ValueError("lateral workspace radius must be positive and contain the full initial X/Y square")
        if not math.isfinite(self.plate_outer_radius_m) or self.plate_outer_radius_m < 0:
            raise ValueError("plate outer radius must be finite and nonnegative; zero selects legacy geometry")
        if not isinstance(self.shape_randomization, bool) or (self.shape_randomization and not (
                0 <= self.rim_probability <= 1
                and 0 < self.rim_height_min_m <= self.rim_height_max_m < self.hole_depth_m
                and 0 < self.rim_width_min_m <= self.rim_width_max_m)):
            raise ValueError("shape randomization requires a valid rim probability, height and width range")
        if self.shape_randomization and self.rough_surface:
            raise ValueError("shape randomization does not support the rough surface terrain")
        if not self.shape_randomization and any(ALL_SHAPES.index(s) >= LEGACY_SHAPE_COUNT for s in self.shapes):
            raise ValueError("rectangle and dual shapes require shape_randomization")
        footprint = self.scale*(math.sqrt(2)*.01 + .035*math.sin(math.radians(35)))
        if (self.plate_outer_radius_m and self.plate_outer_radius_m
                < self.workspace_xy_radius_m + footprint + .001*self.scale):
            raise ValueError("plate must cover workspace plus tilted peg footprint and 1 mm margin")
        if (not math.isfinite(self.initial_z_min) or not math.isfinite(self.initial_z_max)
                or not 0 < self.initial_z_min <= self.initial_z_max < self.workspace_z_max_m * self.scale):
            raise ValueError("initial Z bounds must be positive, ordered and below the workspace ceiling")
        if (not math.isfinite(self.initial_plane_margin)
                or not 0 <= self.initial_plane_margin <= self.initial_z_max):
            raise ValueError("initial plane margin must be finite and within [0, initial_z_max]")
        if (self.rough_surface and self.rough_max_amplitude_m
                + max(self.initial_plane_margin, 0.0005 * self.scale) >= self.initial_z_max):
            raise ValueError("rough_max_amplitude_m plus spawn margin must stay below initial_z_max")
        if (not isinstance(self.noise_zero_steps, int)
                or not isinstance(self.noise_full_steps, int)
                or not 0 <= self.noise_zero_steps < self.noise_full_steps):
            raise ValueError("noise curriculum requires integer steps with 0 <= zero < full")
        if self.noise_curriculum_mode not in ("steps", "success", "adr"):
            raise ValueError("noise curriculum mode must be steps, success or adr")
        if self.noise_curriculum_mode in ("success", "adr") and not self.noise_curriculum:
            raise ValueError("success and ADR noise curricula require noise_curriculum=True")
        if self.training and self.noise_curriculum_mode == "adr" and (
                not self.shape_randomization or {"square", "circle"} & set(self.shapes)):
            raise ValueError("ADR per-episode clearance requires shape randomization without square or circle")
        if self.grasp_error_distribution not in ("gaussian", "uniform"):
            raise ValueError("grasp error distribution must be gaussian or uniform")
        if not math.isfinite(self.target_depth_min_m) or self.target_depth_min_m <= 0:
            raise ValueError("minimum target depth must be finite and positive")
        if not math.isfinite(self.target_depth_max_fraction) or not 0 < self.target_depth_max_fraction < 1:
            raise ValueError("maximum target depth fraction must be strictly between zero and one")
        hole_depth = self.hole_depth_m * self.scale
        if self.fixed_target_depth_m is not None:
            if not math.isfinite(self.fixed_target_depth_m) or not 0 < self.fixed_target_depth_m < hole_depth:
                raise ValueError("fixed target depth must be positive and smaller than hole depth")
        elif self.target_depth_min_m > hole_depth * self.target_depth_max_fraction:
            raise ValueError("minimum target depth exceeds the scaled hole's maximum target depth")
        if not math.isfinite(self.rough_min_amplitude_m) or not 0 <= self.rough_min_amplitude_m <= self.rough_max_amplitude_m:
            raise ValueError("rough_min_amplitude_m must be finite and within [0, rough_max_amplitude_m]")
        if not math.isfinite(self.rough_step_jitter) or not 0 <= self.rough_step_jitter <= 1:
            raise ValueError("rough_step_jitter must be finite and in [0, 1]")
        if not math.isfinite(self.rough_stepped_prob) or not 0 <= self.rough_stepped_prob <= 1:
            raise ValueError("rough_stepped_prob must be finite and in [0, 1]")
        if not math.isfinite(self.rough_low_spawn_prob) or not 0 <= self.rough_low_spawn_prob <= 1:
            raise ValueError("rough_low_spawn_prob must be finite and in [0, 1]")
        if not math.isfinite(self.workspace_z_max_m) or self.workspace_z_max_m <= self.initial_z_max:
            raise ValueError("workspace_z_max_m must be finite and exceed initial_z_max")
        cost_positive = (self.time_cost_per_second, self.position_cost_per_second,
                         self.position_cost_cap, self.xy_cost_radius_fraction,
                         self.orientation_cost_reference_deg, self.load_cost_cap,
                         self.action_cost_cap, self.stagnation_speed_threshold_mps,
                         self.stagnation_window_seconds,
                         self.completion_bonus, self.safety_failure_penalty)
        if any(not math.isfinite(x) or x <= 0 for x in cost_positive):
            raise ValueError("completion cost scales must be finite and positive")
        if self.xy_cost_radius_fraction > 1:
            raise ValueError("xy_cost_radius_fraction must be in (0, 1]")
        if not math.isfinite(self.xy_outer_cost_weight) or not 0 <= self.xy_outer_cost_weight <= 1:
            raise ValueError("xy_outer_cost_weight must be finite and in [0, 1]")
        if not math.isfinite(self.z_outer_cost_weight) or not 0 <= self.z_outer_cost_weight <= 1:
            raise ValueError("z_outer_cost_weight must be finite and in [0, 1]")
        cost_nonnegative = (self.orientation_cost_per_second, self.force_cost_per_second,
                            self.moment_cost_per_second,
                            self.stagnation_cost_per_second, self.surface_z_free_m,
                            self.action_cost_per_second,
                            self.action_change_cost_per_second, self.timeout_penalty,
                            self.force_cost_free_n, self.moment_cost_free_nm,
                            self.insertion_depth_reward)
        if any(not math.isfinite(x) or x < 0 for x in cost_nonnegative):
            raise ValueError("completion cost weights and deadbands must be finite and nonnegative")
        if self.force_cost_free_n >= self.max_force or self.moment_cost_free_nm >= self.max_moment:
            raise ValueError("load cost deadbands must be below safety limits")
        # 卡住时随时可零动作悬停至终止，失败罚分只需超过这条退路的成本即可杜绝自杀
        if self.safety_failure_penalty <= self.hold_cost_bound:
            raise ValueError("safety failure penalty must exceed the discounted cost of holding still until termination")
        if not math.isfinite(self.impedance_walk_fraction) or not 0 <= self.impedance_walk_fraction < 1:
            raise ValueError("impedance walk fraction must be finite and in [0, 1)")
        walk_positive = (self.impedance_walk_rate_per_second,
                         self.impedance_walk_interval_seconds,
                         self.impedance_walk_time_constant_seconds)
        if any(not math.isfinite(x) or x <= 0 for x in walk_positive):
            raise ValueError("impedance walk rates and periods must be finite and positive")
        ticks = self.impedance_walk_interval_seconds / 0.02
        if ticks < 1 or not math.isclose(ticks, round(ticks), abs_tol=1e-8):
            raise ValueError("impedance walk interval must be a positive multiple of 0.02 s")

    @property
    def hold_cost_bound(self) -> float:
        # 零动作、无超限负载时仍不可避免的成本项，按无限期折扣并计入超时罚分
        return self.physics_dt*self.decimation*(
            self.time_cost_per_second + self.position_cost_per_second*self.position_cost_cap
            + 2*self.orientation_cost_per_second + self.stagnation_cost_per_second
        )/(1-PPO_DISCOUNT) + self.timeout_penalty

    @property
    def completion_step_cost_bound(self) -> float:
        # 单步运行成本的理论上界
        load_cap_sq = self.load_cost_cap ** 2
        return self.physics_dt*self.decimation*(
            self.time_cost_per_second + self.position_cost_per_second*self.position_cost_cap
            + 2*self.orientation_cost_per_second + self.force_cost_per_second*load_cap_sq
            + self.moment_cost_per_second*load_cap_sq
            + self.stagnation_cost_per_second
            + self.action_cost_per_second*self.action_cost_cap*math.sqrt(6)
            + self.max_action_change_cost_per_second*2*self.action_cost_cap*math.sqrt(6))

    @property
    def max_action_change_cost_per_second(self) -> float:
        # 平方时按最大跳变幅度折算成线性斜率上界
        air = self.free_air_action_change_cost_per_second or 0.
        if self.free_air_action_change_squared:
            air *= 2*self.action_cost_cap*math.sqrt(6)
        return max(self.action_change_cost_per_second, air)

    @property
    def critic_shape_count(self) -> int:
        return len(ALL_SHAPES) if self.shape_randomization else LEGACY_SHAPE_COUNT

    @property
    def environment_counts(self) -> tuple[int, ...]:
        return self.shape_env_counts or (self.envs_per_shape,) * len(self.shapes)

    @property
    def active_clearance_levels(self) -> tuple[float, ...]:
        # 仅在训练且开启间隙课程时返回从大到小的孔间隙序列
        if not self.training or not self.clearance_curriculum:
            return (self.clearance,)
        higher = tuple(c for c in self.clearance_levels if c > self.clearance + 1e-12)
        return higher + (self.clearance,)

    @property
    def target_depth_bounds_m(self) -> tuple[float, float]:
        if self.fixed_target_depth_m is not None:
            return self.fixed_target_depth_m, self.fixed_target_depth_m
        return self.target_depth_min_m, self.hole_depth_m * self.scale * self.target_depth_max_fraction

    def noise_scale(self, policy_steps: int) -> float:
        if not self.training or not self.noise_curriculum:
            return 1.0
        if self.noise_curriculum_mode in ("success", "adr"):
            return 0.0
        if policy_steps < self.noise_zero_steps:
            return 0.0
        progress = min(1., (policy_steps-self.noise_zero_steps)
                       / (self.noise_full_steps-self.noise_zero_steps))
        return 0.2 + 0.8*progress

    @classmethod
    def from_checkpoint(cls, environment: dict) -> "InsertionCfg":
        values = dict(environment)
        defaults = {
            "position_observation_delay": False,
            "inserted_load_cost": False,
            "force_bias": 0.0,
            "moment_bias": 0.0,
            "rotation_bias": 0.0,
            "grasp_z_error": None,
            "hole_depth_m": HOLE_DEPTH_M,
            "align_goal_z_to_observation": False,
            "stiction_force": 0.0,
            "stiction_moment": 0.0,
            "wall_friction_force": 0.0,
            "load_cost_cap": 1.5,
            "stagnation_cost_per_second": 0.0,
            "stagnation_speed_threshold_mps": 0.006,
            "stagnation_window_seconds": 0.4,
            "surface_z_free_m": 0.0,
            "xy_outer_cost_weight": 0.0,
            "z_outer_cost_weight": 0.0,
            "action_cost_cap": 4.0,
            "terminate_on_overload": False,
            "enable_timeout": True,
            "rough_surface": False,
            "rough_min_amplitude_m": 0.0,
            "rough_max_amplitude_m": 0.0025,
            "rough_wavelength_min_m": 0.008,
            "rough_wavelength_max_m": 0.036,
            "rough_max_slope_deg": 18.0,
            "rough_flat_probability": 0.1,
            "rough_num_waves": 4,
            "rough_step_jitter": 0.0,
            "rough_stepped_prob": 0.0,
            "rough_low_spawn_prob": 0.0,
            "workspace_z_max_m": 0.030,
            "wrench_bias_distribution": "per_axis",
            "wrench_hysteresis_fraction": 0.0,
            "initial_xy": 0.003,
            "workspace_xy_radius_m": 0.025 * values.get("scale", 1.0),
            "plate_outer_radius_m": 0.0,
            "success_opening_required": False,
            "initial_z_max": 0.02,
            "action_change_cost_per_second": 0.05,
            "safety_failure_penalty": 250.0,
            "stiction_force_z": None,
            "stiction_moment_xy": None,
            "mass_min_kg": 0.7,
            "mass_max_kg": 1.5,
            "damping_ratio_min": 0.8,
            "damping_ratio_max": 1.3,
            "down_mass_min_kg": 0.2,
            "down_mass_max_kg": 0.4,
            "plate_friction_min": 0.25,
            "plate_friction_max": 1.15,
            "target_lowpass": False,
            "rotation_pivot_max_m": 0.0,
            "stick_slip_max_pp_m": 0.0,
            "step_bias_rate_per_second": 0.0,
            "free_air_action_change_cost_per_second": None,
            "vibration_force_min_n": 0.0,
            "vibration_force_max_n": 0.0,
            "stiction_reference_stiffness": None,
            "hard_state_reset_probability": 0.0,
            "sensor_tip_min_m": None,
            "target_lowpass_off_probability": 0.0,
        }
        for key, default in defaults.items():
            values.setdefault(key, default)
        # 忽略已退役字段
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in values.items() if key in known}
        if "shapes" in values:
            values["shapes"] = tuple(values["shapes"])
        if "clearance_levels" in values:
            values["clearance_levels"] = tuple(values["clearance_levels"])
        return cls(**values)


def runner_config(steps: int = 256) -> dict:
    cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=steps,
        save_interval=50,
        logger="tensorboard",
        upload_model=False,
        actor=RslRlModelCfg(
            class_name="RNNModel", rnn_type="gru", rnn_hidden_dim=128,
            rnn_num_layers=1, hidden_dims=(128, 64), activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "init_std": 0.4,
                              "std_type": "log", "std_range": (0.02, 1.0)},
        ),
        critic=RslRlModelCfg(
            class_name="RNNModel", rnn_type="gru", rnn_hidden_dim=128,
            rnn_num_layers=2, hidden_dims=(128, 64), activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1e-4, gamma=PPO_DISCOUNT, lam=0.95, clip_param=0.2,
            num_learning_epochs=4, num_mini_batches=4, max_grad_norm=1.0,
            desired_kl=0.008, entropy_coef=0.0,
        ),
    )
    config = asdict(cfg)
    # 悬空力旋量观测平滑正则，抑制策略对自身惯性力/振动的反应
    config["algorithm"].update(class_name="insertion.ppo:SmoothPPO", smooth_coef=0.0125,
                               smooth_force_std=0.5, smooth_moment_std=0.05)
    return config
