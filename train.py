"""统一训练入口：支持有限轮次训练、带守护进程的持续训练以及多卡分布式训练"""

import argparse
import copy
from dataclasses import asdict, replace
import json
import logging
import math
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

import torch
from rsl_rl.runners import OnPolicyRunner

from insertion import InsertionCfg, InsertionEnv, TRAIN_SHAPES, runner_config


def prepare_runner_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    # mjlab 共享模型配置带有 CNN 字段，RNNModel 初始化前需移除
    for key in ("actor", "critic"):
        config[key].pop("cnn_cfg", None)
    return config


def make_runner(env: InsertionEnv, config: dict, log_dir: str | None = None) -> OnPolicyRunner:
    return OnPolicyRunner(env, prepare_runner_config(config), log_dir=log_dir, device=str(env.device))


def contract(cfg: InsertionCfg) -> dict:
    return {
        "version": 1, "actor_dimension": 13, "action_dimension": 6,
        "policy_period_seconds": 0.02, "quaternion_order": "wxyz",
        "physics_cuda_graph": cfg.physics_cuda_graph,
        "observation_cuda_graph": cfg.observation_cuda_graph,
        "position_error": "target minus current, in fixed hole axes",
        "relative_orientation": "inverse(current) * target; canonical unit quaternion",
        "wrench": "gravity-compensated downstream load, in nominal tool axes about nominal tip",
        "inertia_compensated": False,
        "rotation_center": "nominal tool tip",
        "action": "H-frame translation and rotation-vector increment relative to actual tip",
        "translation_limit_m": cfg.translation_limit,
        "rotation_component_limit_rad": cfg.rotation_limit,
        "target_composition": "p_target=p_current+dp; q_target=Exp(dr)*q_current",
        "nominal_body_to_tip_m": [0, 0, -0.05 * cfg.scale],
        "nominal_body_to_sensor_m": [0, 0, 0.03 * cfg.scale],
        "filter_seconds": cfg.filter_seconds,
        "training_noise": {
            "curriculum": {
                "enabled": cfg.noise_curriculum,
                "mode": cfg.noise_curriculum_mode,
                "zero_policy_steps": cfg.noise_zero_steps,
                "full_policy_steps": cfg.noise_full_steps,
                "ramp_initial_scale": 0.2,
                "evaluation_scale": 1.0,
                "fixed_errors_sampling": "scale at episode reset, then fixed for the episode",
                "success_window_episodes_per_shape": 1000,
                "success_threshold_per_shape": 0.8,
                "success_scale_increment": 0.1,
                "success_initial_scale": 0.0,
                "success_statistics": "recent completed episodes at current level; all shapes must pass",
                "clearance_curriculum": cfg.clearance_curriculum,
                "clearance_levels_m": list(cfg.active_clearance_levels),
            },
            "position_std_m": cfg.position_noise,
            "rotation_vector_component_std_rad": cfg.rotation_noise,
            "force_std_n": cfg.force_noise,
            "moment_std_nm": cfg.moment_noise,
            "pose_sampling": "independent Gaussian per observation; current tip pose only",
            "wrench_sampling": "independent Gaussian per observation after tip transform/filter",
            "grasp_distribution": cfg.grasp_error_distribution,
            "grasp_component_scale_m": cfg.grasp_position_error,
            "grasp_scale_semantics": "std" if cfg.grasp_error_distribution == "gaussian" else "half range",
            "grasp_sampling": "per episode",
        },
        "force_limit_n": cfg.max_force,
        "moment_limit_nm": cfg.max_moment,
        "workspace_xy_m": 0.025 * cfg.scale,
        "workspace_z_low_m": -0.025 * cfg.scale,
        "workspace_z_high_m": 0.03 * cfg.scale,
        "max_tilt_rad": 0.6108652381980153,
        "hidden_state": [2, "batch", 128],
        "training_shapes": list(cfg.shapes),
        "target_depth_bounds_m": list(cfg.target_depth_bounds_m),
        "target_depth_sampling": "uniform per episode" if cfg.fixed_target_depth_m is None else "fixed",
        "depth_success": "reach or exceed target, with existing 0.2 mm tolerance",
        "preinsertion_tilt_weight": cfg.preinsertion_tilt_weight,
        "xy_cost_radius_fraction": cfg.xy_cost_radius_fraction,
        "reward_adaptations": (
            "negative dt-scaled running costs; decoupled XY/Z distance; "
            "continuous yaw cost; depth-blended tilt; terminal completion bonus"
        ),
        "timeout_bootstrap": False,
        "impedance_walk": {
            "fraction": cfg.impedance_walk_fraction,
            "rate_per_second": cfg.impedance_walk_rate_per_second,
            "target_interval_seconds": cfg.impedance_walk_interval_seconds,
            "time_constant_seconds": cfg.impedance_walk_time_constant_seconds,
            "nominal_disabled": True,
        },
        "clearance_definition": "opposing-face dimension difference; each wall offset is half",
        "simulation_only": True,
    }


def atomic_json(path: Path, value: dict) -> None:
    # 先写临时文件再原子替换，避免进程中断留下半截 JSON
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def health_state(exit_code: int | None, now: float, started: float,
                 heartbeat: dict | None, stall_seconds: float) -> str:
    if exit_code is not None:
        return "exited"
    last_progress = started if heartbeat is None else max(started, heartbeat["timestamp"])
    return "stalled" if now - last_progress > stall_seconds else "running"


def load_recovery_checkpoint(output: Path, device: str) -> dict:
    checkpoint = output / "checkpoint.pt"
    previous = output / "previous.pt"
    for candidate in (checkpoint, previous):
        if not candidate.exists():
            continue
        try:
            saved = torch.load(candidate, weights_only=True, map_location=device)
            if not isinstance(saved, dict) or not {
                "actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos",
            }.issubset(saved):
                raise ValueError("incomplete recovery checkpoint")
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError, IndexError):
            logging.exception("Cannot read recovery checkpoint %s.", candidate)
            continue
        if candidate == previous:
            logging.warning("Recovering from previous.pt; retaining rejected checkpoint.")
            if checkpoint.exists():
                checkpoint.replace(output / "rejected_checkpoint.pt")
            temporary = output / "checkpoint.pt.tmp"
            shutil.copyfile(previous, temporary)
            temporary.replace(checkpoint)
        return saved
    raise RuntimeError("No readable recovery checkpoint; inspect the explicit errors above.")


class TrainingStopped(Exception):
    pass


def publish_scene(env: InsertionEnv, output: Path, metadata: dict, progress: dict) -> None:
    plants = {}
    for group in env.groups:
        if group.shape in plants:
            continue
        plants[group.shape] = {
            "qpos": group.sim.data.qpos[0].cpu().tolist(),
            "grasp_offset": (group.offset[0]-group.nominal_offset[0]).cpu().tolist(),
            "goal": group.goal[0].cpu().tolist(),
            "wrench": group.wrench[0].cpu().tolist(),
            "stiffness": group.k[0].cpu().tolist(),
            "damping": group.d[0].cpu().tolist(),
            "episode_step": int(group.length[0]),
            "clearance": group.clearance,
            "difficulty": group.difficulty,
            "noise_scale": group.noise_scale,
        }
    atomic_json(output / "scene.json", {
        **progress, "environment": metadata["environment"], "plants": plants,
    })


class LiveInsertionEnv(InsertionEnv):
    def __init__(self, cfg: InsertionCfg, device: str, output: Path, metadata: dict,
                 publish: bool = True):
        simulation_devices = metadata.get("simulation_devices")
        super().__init__(cfg, device, None if simulation_devices is None
                         else tuple(simulation_devices))
        self.output, self.metadata = output, metadata
        self.publish = publish
        self.last_scene_time = 0.

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        now = time.monotonic()
        if self.publish and now-self.last_scene_time >= 0.5:
            publish_scene(self, self.output, self.metadata, {
                "timestamp": time.time(), "policy_steps": self.policy_steps,
            })
            self.last_scene_time = now
        return result


class ContinuousRunner(OnPolicyRunner):
    env: InsertionEnv

    def __init__(self, env: InsertionEnv, config: dict, output: Path, metadata: dict):
        super().__init__(env, prepare_runner_config(config), str(output), str(env.device))
        self.env = env
        self.output, self.metadata = output, metadata
        self.env.curriculum_rollout_steps = config["num_steps_per_env"]
        self.stop_requested = False
        self.last_saved = None

    def save(self, path: str, infos: dict | None = None) -> None:
        del path, infos
        self.env.advance_noise_curriculum()
        for model in (self.alg.actor, self.alg.critic):
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"nonfinite trained parameter: {name}")
        progress_key = (self.current_learning_iteration, self.env.policy_steps)
        if self.last_saved == progress_key:
            if self.stop_requested:
                raise TrainingStopped()
            return
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = {
            **self.metadata,
            "progress": {"policy_steps": self.env.policy_steps,
                         "transitions": self.env.total_transitions},
        }
        if self.env.noise_curriculum is not None:
            saved["infos"]["noise_curriculum_state"] = self.env.noise_curriculum.state_dict()
        temporary = self.output / "checkpoint.pt.tmp"
        checkpoint = self.output / "checkpoint.pt"
        torch.save(saved, temporary)
        if checkpoint.exists():
            previous = self.output / "previous.pt.tmp"
            shutil.copyfile(checkpoint, previous)
            previous.replace(self.output / "previous.pt")
        temporary.replace(checkpoint)
        reward = list(self.logger.rewbuffer)
        memory = self.release_update_cache()
        progress = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "iteration": self.current_learning_iteration,
            "policy_steps": self.env.policy_steps,
            "transitions": self.env.total_transitions,
            "mean_episode_reward": sum(reward)/len(reward) if reward else None,
            "learning_rate": self.alg.learning_rate,
            "clearance": self.env.groups[0].clearance,
            "difficulty": self.env.groups[0].difficulty,
            "noise_scale": self.env.groups[0].noise_scale,
            "checkpoint": str(checkpoint),
            "cuda_memory": memory,
        }
        atomic_json(self.output / "heartbeat.json", progress)
        publish_scene(self.env, self.output, self.metadata, progress)
        self.last_saved = progress_key
        if self.stop_requested:
            raise TrainingStopped()

    def release_update_cache(self) -> dict:
        memory = {}
        for device in dict.fromkeys(group.device for group in self.env.groups):
            with torch.cuda.device(device):
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                # 循环 PPO 的变长序列工作区不应挤占下一轮仿真的显存。
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(device)
                memory[device] = {
                    "allocated_bytes": allocated, "reserved_before_bytes": reserved,
                    "reserved_after_bytes": torch.cuda.memory_reserved(device),
                    "free_bytes": free, "total_bytes": total,
                }
        return memory


def worker(output: Path, device: str, attempt: int) -> None:
    metadata = json.loads((output / "config.json").read_text())
    if metadata.get("collection_backend", "serial") == "distributed":
        from insertion.distributed import run_distributed
        run_distributed(output, device, attempt)
        return
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed + attempt)
    env = LiveInsertionEnv(cfg, device, output, metadata)
    runner = ContinuousRunner(env, metadata["runner"], output, metadata)
    resume_training(runner, cfg, attempt)

    def request_stop(signum, frame):
        del frame
        logging.info("Stop signal %d: saving after the current PPO iteration.", signum)
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while True:
            runner.learn(1000)
            runner.current_learning_iteration += 1
    except TrainingStopped:
        if runner.logger.writer is not None:
            runner.logger.stop_logging_writer()
        logging.info("Stopped with an atomic checkpoint.")


def resume_training(runner: ContinuousRunner, cfg: InsertionCfg, attempt: int,
                    rank: int = 0, world_size: int = 1) -> None:
    if not (runner.output / "checkpoint.pt").exists():
        return
    saved = load_recovery_checkpoint(runner.output, str(runner.env.device))
    restore_training_state(runner, saved)
    runner.current_learning_iteration = saved["iter"]+1
    restore_environment_progress(runner.env, cfg, saved, attempt, rank, world_size)
    logging.info("Resumed iteration %d, policy steps %d; episodes and GRU state reset.",
                 runner.current_learning_iteration, runner.env.policy_steps)


def restore_environment_progress(env: InsertionEnv, cfg: InsertionCfg, saved: dict,
                                 attempt: int, rank: int = 0, world_size: int = 1) -> None:
    env.policy_steps = saved["infos"]["progress"]["policy_steps"]
    source_cfg = saved["infos"]["environment"]
    env.total_transitions = saved["infos"]["progress"].get(
        "transitions", env.policy_steps*source_cfg["envs_per_shape"]*len(source_cfg["shapes"]))
    if env.noise_curriculum is not None:
        if (source_cfg.get("noise_curriculum_mode", "steps") == "success"
                or source_cfg.get("clearance_curriculum", False)):
            if ("noise_curriculum_state" in saved["infos"]
                    and source_cfg.get("clearance_curriculum", False) == cfg.clearance_curriculum):
                env.noise_curriculum.load_state_dict(saved["infos"]["noise_curriculum_state"])
            else:
                logging.info("Resetting success curriculum to level 0 for updated curriculum config.")
        else:
            logging.info("Migrating step-based noise curriculum to success-based level 0.")
        env.curriculum_pending.clear()
    for index, group in enumerate(env.groups):
        if env.noise_curriculum is not None:
            group.difficulty = env.noise_curriculum.difficulty
            group.noise_scale = (env.noise_curriculum.scale
                                 if env.noise_curriculum.noise_levels > 0
                                 else cfg.noise_scale(env.policy_steps))
            if hasattr(group, "set_clearance"):
                group.set_clearance(env.noise_curriculum.clearance)
        else:
            group.difficulty = min(1., 0.25 + 0.75*env.policy_steps/cfg.curriculum_steps)
            group.noise_scale = cfg.noise_scale(env.policy_steps)
        seed = cfg.seed + (index*world_size + rank)*10007 + env.policy_steps + attempt
        group.generator.manual_seed(seed)
        group.walk_generator.manual_seed(seed + 500003)
        group.reset(torch.arange(group.n, device=group.device))


def restore_training_state(runner: ContinuousRunner, saved: dict,
                           cfg: InsertionCfg | None = None) -> None:
    del cfg
    runner.alg.load(saved, None, True)


def registered_worker_pid(output: Path) -> int:
    pid = json.loads((output / "processes.json").read_text())["worker_pid"]
    command = Path(f"/proc/{pid}/cmdline")
    if not command.exists():
        raise ProcessLookupError(f"registered worker PID {pid} is no longer running; retry after recovery")
    arguments = command.read_bytes().decode().split("\0")
    if "--worker" not in arguments or str(output) not in arguments:
        raise RuntimeError("registered PID is not this run's training worker; refusing to signal it")
    return pid


def resize_training(output: Path, envs_per_shape: int) -> None:
    output = output.resolve()
    metadata = json.loads((output / "config.json").read_text())
    cfg = replace(InsertionCfg.from_checkpoint(metadata["environment"]),
                  envs_per_shape=envs_per_shape)
    devices = metadata.get("simulation_devices", ["cuda:0"])
    if envs_per_shape % len(devices) or envs_per_shape*len(cfg.shapes) % 4:
        raise ValueError("environment counts must divide evenly across devices and PPO mini-batches")
    if metadata.get("collection_backend", "serial") == "distributed":
        from insertion.distributed import validate_layout
        validate_layout({**metadata, "environment": asdict(cfg)}, "cuda:0")
    pid = registered_worker_pid(output)
    timestamp = time.time_ns()
    atomic_json(output / f"config_before_resize_{timestamp}.json", metadata)
    atomic_json(output / f"resize_{timestamp}.json", {
        "timestamp": time.time(), "worker_pid": pid,
        "from_envs_per_shape": metadata["environment"]["envs_per_shape"],
        "to_envs_per_shape": envs_per_shape,
        "retained": ["actor", "critic", "normalization", "optimizer", "curriculum"],
    })
    metadata.update(environment=asdict(cfg), contract=contract(cfg))
    atomic_json(output / "config.json", metadata)
    logging.warning("Resize prepared; requesting a graceful handoff from worker PID %d.", pid)
    os.kill(pid, signal.SIGTERM)


def stop_worker(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            logging.error("Worker PID %d exceeded stop grace; killing this child only.", process.pid)
            process.kill()
            process.wait()


def supervise(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        cfg = InsertionCfg(shapes=tuple(args.shapes), envs_per_shape=args.envs_per_shape,
                           seed=args.seed, episode_seconds=args.episode_seconds,
                           clearance_curriculum=args.clearance_curriculum,
                           noise_curriculum=args.noise_curriculum or args.success_noise_curriculum,
                           noise_curriculum_mode="success" if args.success_noise_curriculum else "steps",
                           physics_cuda_graph=args.physics_cuda_graph,
                           observation_cuda_graph=args.observation_cuda_graph,
                           noise_zero_steps=100*args.steps, noise_full_steps=800*args.steps,
                           target_depth_min_m=args.target_depth_min,
                           target_depth_max_fraction=args.target_depth_max_fraction)
        if cfg.envs_per_shape * len(cfg.shapes) % 4:
            raise ValueError("environment count must be divisible by four")
        config = runner_config(args.steps)
        config["save_interval"] = 1
        atomic_json(output / "config.json", {
            "environment": asdict(cfg), "runner": config, "contract": contract(cfg),
            "simulation_devices": args.simulation_devices or [args.device],
            "collection_backend": "distributed" if args.distributed_ppo else "serial",
        })
    elif not (output / "config.json").is_file():
        raise FileNotFoundError("resume requires an existing run config.json")
    metadata = json.loads((output / "config.json").read_text())
    if metadata.get("collection_backend", "serial") == "distributed":
        from insertion.distributed import validate_layout
        validate_layout(metadata, args.device)
    elif metadata.get("collection_backend", "serial") != "serial":
        raise ValueError("unknown collection backend")
    if (output / "STOP").exists():
        raise RuntimeError("STOP exists; remove this run's STOP file before explicitly resuming")
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        del frame
        stopping = True
        logging.info("Supervisor received stop signal %d.", signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.time()
    next_check = start + args.check_seconds
    attempt = 0
    process = None
    with (output / "supervisor.jsonl").open("a", buffering=1) as events:
        def record(kind: str, **details) -> None:
            event = {"timestamp": time.time(), "kind": kind, **details}
            events.write(json.dumps(event, allow_nan=False) + "\n")
            logging.info("%s", event)

        try:
            while not stopping and not (output / "STOP").exists():
                with (output / "training.log").open("a", buffering=1) as log:
                    started = time.time()
                    process = subprocess.Popen([
                        sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                        "--output", str(output), "--device", args.device,
                        "--attempt", str(attempt),
                    ], stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(output / "processes.json", {
                        "supervisor_pid": os.getpid(), "worker_pid": process.pid,
                        "attempt": attempt, "started": started,
                    })
                    record("started", pid=process.pid, attempt=attempt)
                    while not stopping and not (output / "STOP").exists():
                        heartbeat_path = output / "heartbeat.json"
                        heartbeat = (json.loads(heartbeat_path.read_text())
                                     if heartbeat_path.exists() else None)
                        now = time.time()
                        state = health_state(process.poll(), now, started, heartbeat, args.stall_seconds)
                        if now >= next_check:
                            record("scheduled_check", state=state, pid=process.pid,
                                   elapsed_seconds=now-start, heartbeat=heartbeat)
                            next_check += args.check_seconds
                        if state != "running":
                            record("anomaly", state=state, exit_code=process.poll(),
                                   pid=process.pid, heartbeat=heartbeat)
                            stop_worker(process, args.stop_grace_seconds)
                            break
                        time.sleep(3)
                    if stopping or (output / "STOP").exists():
                        stop_worker(process, args.stop_grace_seconds)
                        record("stopped", pid=process.pid, exit_code=process.poll())
                        break
                attempt += 1
                record("restarting", attempt=attempt, delay_seconds=30,
                       checkpoint_exists=(output / "checkpoint.pt").exists())
                for _ in range(30):
                    if stopping or (output / "STOP").exists():
                        break
                    time.sleep(1)
        finally:
            if process is not None:
                stop_worker(process, args.stop_grace_seconds)


def run_finite(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg = InsertionCfg(shapes=tuple(args.shapes), envs_per_shape=args.envs_per_shape,
                       seed=args.seed, episode_seconds=args.episode_seconds,
                       clearance_curriculum=args.clearance_curriculum,
                       noise_curriculum=args.noise_curriculum or args.success_noise_curriculum,
                       noise_curriculum_mode="success" if args.success_noise_curriculum else "steps",
                       physics_cuda_graph=args.physics_cuda_graph,
                       observation_cuda_graph=args.observation_cuda_graph,
                       noise_zero_steps=100*args.steps, noise_full_steps=800*args.steps,
                       target_depth_min_m=args.target_depth_min,
                       target_depth_max_fraction=args.target_depth_max_fraction)
    count = cfg.envs_per_shape * len(cfg.shapes)
    if count < 4 or count % 4:
        raise ValueError("environment count must be a positive multiple of four")
    args.output.mkdir(parents=True, exist_ok=False)
    config = runner_config(args.steps)
    metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg),
                "simulation_devices": args.simulation_devices or [args.device],
                "collection_backend": "distributed" if args.distributed_ppo else "serial"}
    atomic_json(args.output / "config.json", metadata)
    if args.distributed_ppo:
        from insertion.distributed import run_distributed
        run_distributed(args.output.resolve(), args.device, iterations=args.iterations)
        return
    env = InsertionEnv(cfg, args.device,
                       None if args.simulation_devices is None else tuple(args.simulation_devices))
    runner = make_runner(env, config, str(args.output))
    runner.learn(args.iterations)
    runner.save(str(args.output / "checkpoint.pt"), infos=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/insertion"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--simulation-devices", nargs="+",
                        help="distribute geometry groups across these devices; PPO uses --device")
    parser.add_argument("--distributed-ppo", action="store_true",
                        help="one local collector/PPO worker per GPU, with synchronized gradients")
    parser.add_argument("--iterations", type=int, default=None,
                        help="run a fixed number of PPO iterations; omit for continuous supervised training")
    parser.add_argument("--envs-per-shape", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--target-depth-min", type=float, default=0.005, help="metres")
    parser.add_argument("--target-depth-max-fraction", type=float, default=0.9)
    parser.add_argument("--shapes", nargs="+", choices=TRAIN_SHAPES, default=list(TRAIN_SHAPES))
    parser.add_argument("--noise-curriculum", action="store_true")
    parser.add_argument("--success-noise-curriculum", action="store_true")
    parser.add_argument("--clearance-curriculum",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physics-cuda-graph", action="store_true")
    parser.add_argument("--observation-cuda-graph", action="store_true")
    parser.add_argument("--check-seconds", type=float, default=300)
    parser.add_argument("--stall-seconds", type=float, default=900)
    parser.add_argument("--stop-grace-seconds", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resize-envs", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 4 or args.envs_per_shape < 1:
        parser.error("rollout steps must be >= 4 and environments positive")
    if args.iterations is not None and args.iterations < 1:
        parser.error("iterations must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.check_seconds, args.stall_seconds, args.stop_grace_seconds)):
        parser.error("monitor intervals must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.worker:
        worker(args.output, args.device, args.attempt)
    elif args.resize_envs:
        resize_training(args.output, args.envs_per_shape)
    elif args.iterations is not None and not args.resume:
        run_finite(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
