"""多 GPU 分布式并行采样与同步循环 PPO 训练"""

import argparse
import copy
from dataclasses import replace
import json
import logging
from multiprocessing.synchronize import Event
from pathlib import Path
import os
import signal
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessContext, spawn
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils import check_nan

from .config import InsertionCfg
from .env import InsertionEnv
from train import (ContinuousRunner, LiveInsertionEnv, TrainingStopped, atomic_json,
                   load_recovery_checkpoint, resume_training)


def validate_layout(metadata: dict, device: str) -> int:
    devices = metadata.get("simulation_devices", [device])
    world_size = len(devices)
    if world_size < 2 or devices != [f"cuda:{i}" for i in range(world_size)] or device != "cuda:0":
        raise ValueError("distributed PPO requires cuda:0 as primary and contiguous CUDA devices")
    if world_size > torch.cuda.device_count():
        raise ValueError("distributed PPO requests unavailable CUDA devices")
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    if (cfg.envs_per_shape % world_size
            or any(n % world_size for n in cfg.environment_counts)
            or sum(cfg.environment_counts)//world_size % 4):
        raise ValueError("each rank must have whole shape shards and an environment count divisible by four")
    if metadata["runner"]["algorithm"]["class_name"] != "PPO" or any(
            metadata["runner"][key]["class_name"] != "RNNModel" for key in ("actor", "critic")):
        raise ValueError("distributed insertion training requires the existing recurrent PPO configuration")
    return world_size


def gather_batch(value: torch.Tensor) -> torch.Tensor:
    gathered = torch.empty((value.shape[0]*dist.get_world_size(), *value.shape[1:]),
                           device=value.device, dtype=value.dtype)
    dist.all_gather_single(gathered, value.contiguous())
    return gathered


class SharedRNNModel(RNNModel):
    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            if not isinstance(self.obs_normalizer, EmpiricalNormalization):
                raise TypeError("expected the repository's empirical observation normalizer")
            values = torch.cat([obs[key] for key in self.obs_groups], dim=-1)
            self.obs_normalizer.update(gather_batch(values))


def global_advantages(raw: torch.Tensor) -> torch.Tensor:
    values = raw.double()
    stats = torch.stack((values.sum(), values.square().sum(),
                         values.new_tensor(values.numel())))
    dist.all_reduce(stats)
    mean = stats[0]/stats[2]
    variance = ((stats[1] - stats[0]*mean)/(stats[2]-1)).clamp_min(0)
    return (raw-mean.to(raw.dtype))/(variance.sqrt().to(raw.dtype)+1e-8)


class SharedPPO(PPO):
    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages = global_advantages(self.storage.returns-self.storage.values)


class DistributedInsertionEnv(LiveInsertionEnv):
    def __init__(self, cfg: InsertionCfg, rank: int, world_size: int, output: Path, metadata: dict):
        local_cfg = replace(cfg, envs_per_shape=cfg.envs_per_shape//world_size,
                            shape_env_counts=tuple(n//world_size for n in cfg.environment_counts),
                            seed=cfg.seed+rank*1000003)
        device = f"cuda:{rank}"
        super().__init__(local_cfg, device, output,
                         {**metadata, "simulation_devices": [device]}, publish=rank == 0)
        self.metadata, self.world_size = metadata, world_size
        self.curriculum_distributed = True

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        self.total_transitions += self.num_envs*(self.world_size-1)
        return result


def global_logging_step(env: InsertionEnv, rewards: torch.Tensor, dones: torch.Tensor,
                        extras: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    fields = [rewards, dones.float()]
    if env.noise_curriculum is not None:
        fields.extend(((extras["terminal"]["reason"] == 1).float(),
                       extras["terminal"]["curriculum_eligible"].float()))
    outcomes = gather_batch(torch.stack(fields, dim=1))
    if env.noise_curriculum is not None:
        shape_ids = env.curriculum_shape_ids.repeat(dist.get_world_size())
        env.record_curriculum_outcomes(torch.stack(
            (shape_ids, outcomes[:, 1], outcomes[:, 2], outcomes[:, 3]), dim=1))
    keys = [key for key in extras["log"] if not key.startswith("success/")]
    counts = []
    offset = 0
    for group in env.groups:
        ended = dones[offset:offset+group.n]
        reasons = extras["terminal"]["reason"][offset:offset+group.n]
        counts.extend(((ended & (reasons == 1)).float().sum(), ended.float().sum()))
        offset += group.n
    stats = torch.stack([extras["log"][key] for key in keys]+counts)
    dist.all_reduce(stats)
    logs = {key: stats[i]/dist.get_world_size() for i, key in enumerate(keys)}
    for i, group in enumerate(env.groups):
        successes, episodes = stats[len(keys)+2*i:len(keys)+2*i+2]
        if env.noise_curriculum is None and episodes > 0:
            logs[f"success/{group.shape}"] = successes/episodes
    return outcomes[:, 0], outcomes[:, 1].bool(), {"log": logs}


class DistributedRunner(ContinuousRunner):
    env: DistributedInsertionEnv

    def __init__(self, env: DistributedInsertionEnv, metadata: dict, output: Path):
        metadata = {**metadata, "collection_backend": "distributed"}
        config = copy.deepcopy(metadata["runner"])
        config["algorithm"]["class_name"] = "insertion.distributed:SharedPPO"
        for key in ("actor", "critic"):
            config[key]["class_name"] = "insertion.distributed:SharedRNNModel"
        super().__init__(env, config, output, metadata)
        self.env = env
        size = env.num_envs*self.gpu_world_size
        self.logger.cur_reward_sum = torch.zeros(size, device=env.device)
        self.logger.cur_episode_length = torch.zeros(size, device=env.device)

    def learn_parallel(self, stop_event: Event, iterations: int | None = None,
                       verify: bool = False) -> None:
        self.alg.train_mode()
        self.alg.broadcast_parameters()
        obs = self.env.get_observations()
        self.logger.init_logging_writer()
        first = self.current_learning_iteration
        iteration = first
        try:
            while iterations is None or iteration < first+iterations:
                start = time.perf_counter()
                iteration_start = start
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(actions)
                        check_nan(obs, rewards, dones)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        log_rewards, log_dones, log_extras = global_logging_step(
                            self.env, rewards, dones, extras)
                        self.logger.process_env_step(log_rewards, log_dones, log_extras)
                    torch.cuda.synchronize(self.device)
                    collection = time.perf_counter()-start
                    start = time.perf_counter()
                    self.alg.compute_returns(obs)
                losses = self.alg.update()
                torch.cuda.synchronize(self.device)
                timing = torch.tensor([collection, time.perf_counter()-start], device=self.device)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                self.current_learning_iteration = iteration
                curriculum_logs = self.env.advance_noise_curriculum()
                if curriculum_logs:
                    self.logger.ep_extras.append(curriculum_logs)
                self.logger.log(it=iteration, start_it=first,
                                total_it=first+iterations if iterations is not None else iteration+1000,
                                collect_time=float(timing[0]), learn_time=float(timing[1]),
                                loss_dict=losses, learning_rate=self.alg.learning_rate,
                                action_std=self.alg.actor.output_std, rnd_weight=None)
                sys.stdout.flush()
                if verify:
                    self.verify_shared_state()
                if self.gpu_global_rank == 0:
                    self.save(str(self.output / "checkpoint.pt"))
                else:
                    self.release_update_cache()
                # 等待主进程完成原子保存后再进入下一轮采样
                dist.barrier()
                stopping = torch.tensor(int(stop_event.is_set() or (self.output/"STOP").exists()),
                                        device=self.device)
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
                complete_time = torch.tensor(time.perf_counter()-iteration_start, device=self.device)
                dist.all_reduce(complete_time, op=dist.ReduceOp.MAX)
                if self.gpu_global_rank == 0:
                    logging.info("Iteration %d | Complete iteration time (including checkpoint/barrier): %.3fs",
                                 iteration, float(complete_time))
                    if self.logger.writer is not None:
                        self.logger.writer.add_scalar(
                            "Perf/complete_iteration_time", float(complete_time), iteration)
                if stopping:
                    raise TrainingStopped()
                iteration += 1
        finally:
            if self.logger.writer is not None:
                self.logger.stop_logging_writer()

    def verify_shared_state(self) -> None:
        if self.env.noise_curriculum is not None:
            state = self.env.noise_curriculum.state_dict()
            values = [state["level"]]
            for window in state["windows"].values():
                values.extend((len(window), *window))
            actual = torch.tensor(values, device=self.device)
            expected = actual.clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for model in (self.alg.actor, self.alg.critic):
            for value in model.state_dict().values():
                expected = value.clone()
                dist.broadcast(expected, src=0)
                torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-6)
        for state in self.alg.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    actual = value.to(self.device)
                    expected = actual.clone()
                    dist.broadcast(expected, src=0)
                    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        learning_rate = torch.tensor(self.alg.learning_rate, device=self.device)
        rates = gather_batch(learning_rate.reshape(1))
        torch.testing.assert_close(rates, rates[0].expand_as(rates))
        raw = torch.arange(12, device=self.device, dtype=torch.float32)+100*self.gpu_global_rank
        combined = gather_batch(raw)
        expected = (combined-combined.mean())/(combined.std()+1e-8)
        offset = self.gpu_global_rank*len(raw)
        torch.testing.assert_close(global_advantages(raw), expected[offset:offset+len(raw)])


def verify_physics(env: InsertionEnv) -> None:
    reference = InsertionEnv(InsertionCfg.from_checkpoint(env.cfg), str(env.device))
    reference.get_observations()  # 对齐训练器启动时的首次观测采样状态
    for step in range(60):
        actions = torch.zeros(env.num_envs, 6, device=env.device)
        actions[:, 2] = -1
        actions[:, 0] = 0.1 if step % 2 else -0.1
        actual = env.step(actions)
        expected = reference.step(actions)
        for key in ("actor", "critic"):
            torch.testing.assert_close(actual[0][key], expected[0][key], atol=1e-7, rtol=1e-5,
                                       msg=lambda message: f"step {step}, {key}: {message}")
        for i in (1, 2):
            torch.testing.assert_close(actual[i], expected[i], atol=1e-7, rtol=1e-5)
        for key in expected[3]["terminal"]:
            torch.testing.assert_close(actual[3]["terminal"][key], expected[3]["terminal"][key],
                                       atol=1e-7, rtol=1e-5)
        for group, other in zip(env.groups, reference.groups, strict=True):
            for value in ("filtered_sensor", "peak_force", "peak_moment", "walk_current"):
                torch.testing.assert_close(getattr(group, value), getattr(other, value),
                                           atol=1e-7, rtol=1e-5)


def configure_rank(rank: int, world_size: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
                      RANK=str(rank), LOCAL_RANK=str(rank))
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s")
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)


def rank_worker(rank: int, world_size: int, port: int, output: Path, attempt: int,
                stop_event: Event, iterations: int | None, verify: bool) -> None:
    configure_rank(rank, world_size, port)
    metadata = json.loads((output / "config.json").read_text())
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    torch.manual_seed(cfg.seed+attempt+rank)
    env = DistributedInsertionEnv(cfg, rank, world_size, output, metadata)
    runner = DistributedRunner(env, metadata, output)
    try:
        if verify:
            verify_physics(env)
            env = DistributedInsertionEnv(cfg, rank, world_size, output, metadata)
            runner.env = env
        resume_training(runner, cfg, attempt, rank, world_size)
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        runner.learn_parallel(stop_event, iterations, verify)
    except TrainingStopped:
        logging.info("Stopped with a shared atomic checkpoint.")
    finally:
        dist.destroy_process_group()


def run_distributed(output: Path, device: str, attempt: int = 0,
                    iterations: int | None = None, verify: bool = False) -> None:
    metadata = json.loads((output / "config.json").read_text())
    world_size = validate_layout(metadata, device)
    if (output/"checkpoint.pt").exists() or (output/"previous.pt").exists():
        # 启动多卡进程前先修复可能损坏的最新检查点，避免各卡并发冲突
        load_recovery_checkpoint(output, "cpu")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    stop_event = mp.get_context("spawn").Event()
    stopped_at = None

    def request_stop(signum, frame):
        nonlocal stopped_at
        del frame
        logging.info("Distributed coordinator received signal %d; finish and save this iteration.", signum)
        stop_event.set()
        if stopped_at is None:
            stopped_at = time.monotonic()

    previous_handlers = {sig: signal.signal(sig, request_stop)
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    context: ProcessContext | None = None
    try:
        arguments = (world_size, port, output, attempt, stop_event, iterations, verify)
        context = spawn(rank_worker, args=arguments, nprocs=world_size, join=False)
        assert context is not None
        atomic_json(output / "collectors.json", {
            "coordinator_pid": os.getpid(), "rank_pids": [p.pid for p in context.processes],
            "devices": metadata["simulation_devices"], "started": time.time(),
        })
        while not context.join(timeout=1):
            if stopped_at is not None and time.monotonic()-stopped_at > 120:
                raise TimeoutError("distributed workers did not finish the requested checkpoint within 120 seconds")
    finally:
        if context is not None:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=10)
                if process.is_alive():
                    logging.error("Killing unresponsive owned collector PID %d.", process.pid)
                    process.kill()
                    process.join()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    run_distributed(args.output.resolve(), "cuda:0", iterations=args.iterations,
                    verify=args.verify)
