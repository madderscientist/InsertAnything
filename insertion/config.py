"""仿真与训练参数配置，仅用于物理仿真环境"""

from dataclasses import asdict, dataclass
import math

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

TRAIN_SHAPES = ("triangle", "square", "pentagon", "hexagon", "l")
ALL_SHAPES = TRAIN_SHAPES + ("circle",)
HOLE_DEPTH_M = 0.025
PPO_DISCOUNT = 0.9985
CLEARANCE_CURRICULUM_LEVELS = (0.002, 0.001, 0.0005, 0.00025, 0.00015, 0.0001)


@dataclass
class InsertionCfg:
    shapes: tuple[str, ...] = TRAIN_SHAPES
    envs_per_shape: int = 16
    seed: int = 42
    physics_dt: float = 0.001
    decimation: int = 20
    physics_cuda_graph: bool = False
    observation_cuda_graph: bool = False
    episode_seconds: float = 8.0
    clearance: float = 0.0001
    clearance_curriculum: bool = False
    clearance_levels: tuple[float, ...] = CLEARANCE_CURRICULUM_LEVELS
    scale: float = 1.0
    target_depth_min_m: float = 0.005
    target_depth_max_fraction: float = 0.9
    fixed_target_depth_m: float | None = None
    translation_limit: float = 0.001
    rotation_limit: float = math.radians(0.5)
    max_force: float = 30.0
    max_moment: float = 2.0
    filter_seconds: float = 0.015
    initial_xy: float = 0.003
    initial_tilt: float = math.radians(3)
    initial_yaw: float = math.radians(15)
    position_bias: float = 0.0005
    tip_calibration_error: float = 0.0005
    grasp_position_error: float = 0.00125
    grasp_error_distribution: str = "gaussian"
    sensor_rotation_error: float = math.radians(0.5)
    position_noise: float = 0.0005
    rotation_noise: float = math.radians(1.5)
    force_noise: float = 0.2
    moment_noise: float = 0.001
    noise_curriculum: bool = False
    noise_curriculum_mode: str = "steps"
    noise_zero_steps: int = 25600
    noise_full_steps: int = 204800
    dynamics: str = "train"
    training: bool = True
    success_hold_steps: int = 5
    curriculum_steps: int = 256000
    impedance_walk_fraction: float = 0.1
    impedance_walk_rate_per_second: float = 0.05
    impedance_walk_interval_seconds: float = 1.0
    impedance_walk_time_constant_seconds: float = 1.0
    time_cost_per_second: float = 1.0
    position_cost_per_second: float = 2.0
    position_cost_cap: float = 2.0
    xy_cost_radius_fraction: float = 0.25
    orientation_cost_per_second: float = 1.0
    orientation_cost_reference_deg: float = 5.0
    preinsertion_tilt_weight: float = 0.25
    force_cost_per_second: float = 1.0
    moment_cost_per_second: float = 1.0
    force_cost_free_n: float = 5.0
    moment_cost_free_nm: float = 0.1
    action_cost_per_second: float = 0.05
    action_change_cost_per_second: float = 0.1
    completion_bonus: float = 20.0
    safety_failure_penalty: float = 200.0
    timeout_penalty: float = 10.0

    def __post_init__(self) -> None:
        self.shapes = tuple(self.shapes)
        if not self.shapes or any(s not in ALL_SHAPES for s in self.shapes):
            raise ValueError(f"shapes must belong to {ALL_SHAPES}")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("duplicate geometry groups are not allowed")
        if self.envs_per_shape < 1 or self.decimation < 1:
            raise ValueError("environment count and decimation must be positive")
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
        if self.dynamics not in ("train", "nominal", "soft", "stiff"):
            raise ValueError("unknown dynamics profile")
        if self.success_hold_steps < 1 or self.curriculum_steps < 1:
            raise ValueError("success hold and curriculum horizon must be positive")
        noise = (self.initial_xy, self.initial_tilt, self.initial_yaw, self.position_bias,
                 self.tip_calibration_error, self.grasp_position_error, self.sensor_rotation_error,
                 self.position_noise, self.rotation_noise, self.force_noise, self.moment_noise)
        if any(x < 0 or not math.isfinite(x) for x in noise):
            raise ValueError("randomization magnitudes must be finite and nonnegative")
        if (not isinstance(self.noise_zero_steps, int)
                or not isinstance(self.noise_full_steps, int)
                or not 0 <= self.noise_zero_steps < self.noise_full_steps):
            raise ValueError("noise curriculum requires integer steps with 0 <= zero < full")
        if self.noise_curriculum_mode not in ("steps", "success"):
            raise ValueError("noise curriculum mode must be steps or success")
        if self.noise_curriculum_mode == "success" and not self.noise_curriculum:
            raise ValueError("success noise curriculum requires noise_curriculum=True")
        if self.grasp_error_distribution not in ("gaussian", "uniform"):
            raise ValueError("grasp error distribution must be gaussian or uniform")
        if not math.isfinite(self.target_depth_min_m) or self.target_depth_min_m <= 0:
            raise ValueError("minimum target depth must be finite and positive")
        if not math.isfinite(self.target_depth_max_fraction) or not 0 < self.target_depth_max_fraction < 1:
            raise ValueError("maximum target depth fraction must be strictly between zero and one")
        hole_depth = HOLE_DEPTH_M * self.scale
        if self.fixed_target_depth_m is not None:
            if not math.isfinite(self.fixed_target_depth_m) or not 0 < self.fixed_target_depth_m < hole_depth:
                raise ValueError("fixed target depth must be positive and smaller than hole depth")
        elif self.target_depth_min_m > hole_depth * self.target_depth_max_fraction:
            raise ValueError("minimum target depth exceeds the scaled hole's maximum target depth")
        cost_positive = (self.time_cost_per_second, self.position_cost_per_second,
                         self.position_cost_cap, self.xy_cost_radius_fraction,
                         self.orientation_cost_reference_deg,
                         self.completion_bonus, self.safety_failure_penalty)
        if any(not math.isfinite(x) or x <= 0 for x in cost_positive):
            raise ValueError("completion cost scales must be finite and positive")
        if self.xy_cost_radius_fraction > 1:
            raise ValueError("xy_cost_radius_fraction must be in (0, 1]")
        if not math.isfinite(self.preinsertion_tilt_weight) or not 0 <= self.preinsertion_tilt_weight <= 1:
            raise ValueError("preinsertion_tilt_weight must be finite and in [0, 1]")
        cost_nonnegative = (self.orientation_cost_per_second, self.force_cost_per_second,
                            self.moment_cost_per_second, self.action_cost_per_second,
                            self.action_change_cost_per_second, self.timeout_penalty,
                            self.force_cost_free_n, self.moment_cost_free_nm)
        if any(not math.isfinite(x) or x < 0 for x in cost_nonnegative):
            raise ValueError("completion cost weights and deadbands must be finite and nonnegative")
        if self.force_cost_free_n >= self.max_force or self.moment_cost_free_nm >= self.max_moment:
            raise ValueError("load cost deadbands must be below safety limits")
        # 安全失败罚分必须严格大于最差折扣累计运行成本，防止智能体通过自杀提前终止逃避惩罚
        if self.safety_failure_penalty <= max(
                self.completion_step_cost_bound/(1-PPO_DISCOUNT),
                self.completion_step_cost_bound+self.timeout_penalty):
            raise ValueError("safety failure penalty must exceed the worst discounted running/deadline cost")
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
    def completion_step_cost_bound(self) -> float:
        # 单步运行成本的理论上界，用于校验安全失败罚分下界
        return self.physics_dt*self.decimation*(
            self.time_cost_per_second + self.position_cost_per_second*self.position_cost_cap
            + 2*self.orientation_cost_per_second + self.force_cost_per_second
            + self.moment_cost_per_second + self.action_cost_per_second*math.sqrt(6)
            + self.action_change_cost_per_second*2*math.sqrt(6))

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
        return self.target_depth_min_m, HOLE_DEPTH_M * self.scale * self.target_depth_max_fraction

    def noise_scale(self, policy_steps: int) -> float:
        if not self.training or not self.noise_curriculum:
            return 1.0
        if self.noise_curriculum_mode == "success":
            return 0.0
        if policy_steps < self.noise_zero_steps:
            return 0.0
        progress = min(1., (policy_steps-self.noise_zero_steps)
                       / (self.noise_full_steps-self.noise_zero_steps))
        return 0.2 + 0.8*progress

    @classmethod
    def from_checkpoint(cls, environment: dict) -> "InsertionCfg":
        values = dict(environment)
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
            rnn_num_layers=2, hidden_dims=(128, 64), activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.4,
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
    return asdict(cfg)
