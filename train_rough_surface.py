"""第二阶段专用训练入口：随机空间频率/幅度的凹凸不平上表面 + 无超时自主脱困持续训练"""

import argparse
import copy
from dataclasses import asdict, replace
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO

from insertion import InsertionCfg, TRAIN_SHAPES
from insertion.transforms import rotate, tip_pose
from train import (ContinuousRunner, LiveInsertionEnv, TrainingStopped,
                   atomic_json, contract, health_state, resume_training,
                   stop_worker)


class RoughLiveInsertionEnv(LiveInsertionEnv):
    """在标准训练环境基础上实时监控多贴片接触缓冲水位与长回合脱困统计"""

    def __init__(self, cfg: InsertionCfg, device: str, output: Path, metadata: dict,
                 publish: bool = True):
        import warp as wp

        super().__init__(cfg, device, output, metadata, publish=publish)
        self.insertion_cfg = cfg
        self.nacon_views = {
            g.shape: wp.to_torch(g.sim.wp_data.nacon) for g in self.groups}
        self.nefc_views = {
            g.shape: wp.to_torch(g.sim.wp_data.nefc) for g in self.groups}
        self.nacon_limits = {
            g.shape: int(g.sim.wp_data.naconmax) for g in self.groups}
        self.njmax_limits = {
            g.shape: int(g.sim.wp_data.njmax) for g in self.groups}
        self.peak_nacon = {
            s: torch.zeros(1, dtype=torch.int64, device=self.device) for s in cfg.shapes}
        self.peak_nefc = {
            s: torch.zeros(1, dtype=torch.int64, device=self.device) for s in cfg.shapes}
        self.contact_overflow_steps = {
            s: torch.zeros(1, dtype=torch.int64, device=self.device) for s in cfg.shapes}
        self.escaped_stuck_counts = {
            s: torch.zeros(3, dtype=torch.int64, device=self.device) for s in cfg.shapes}
        self.rim_stall_steps = {
            g.shape: torch.zeros(g.n, dtype=torch.int64, device=self.device) for g in self.groups}

    @torch.no_grad()
    def current_wedge_gate(self) -> torch.Tensor:
        cfg = self.insertion_cfg
        gates = []
        for group in self.groups:
            pos, q, _, _ = group.state()
            tip = tip_pose(pos, q, group.offset)
            xy = torch.linalg.vector_norm(tip[:, :2], dim=-1)
            lowest_z = tip[:, 2] + group.minimum_peg_height_offset(q)
            if cfg.inserted_load_cost and cfg.inserted_load_depth_m > 0:
                z_weight = (1.0 + tip[:, 2] / cfg.inserted_load_depth_m).clamp(0.0, 1.0)
            else:
                z_weight = (tip[:, 2] >= 0.0).to(tip.dtype)
            not_at_goal = (tip[:, 2] > group.goal[:, 2] + 0.0005).to(tip.dtype)
            prog_pos = torch.cat((tip[:, :2], tip[:, 2:3].clamp_max(0.0)), dim=-1)
            eff_speed = torch.linalg.vector_norm(prog_pos - group.tip_ema, dim=-1) / cfg.stagnation_window_seconds
            stalled = (1.0 - eff_speed / cfg.stagnation_speed_threshold_mps).clamp(0.0, 1.0).square()
            outside_rim = ((xy > 0.0012 * cfg.scale) | (lowest_z > 0.0005 * cfg.scale)).to(tip.dtype)
            warmup = ((group.length.to(tip.dtype) * 0.02 - 0.6) / 0.3).clamp(0.0, 1.0)
            rim_deadlock = ((self.rim_stall_steps[group.shape].to(tip.dtype) - 110.0) / 25.0).clamp(0.0, 1.0)
            spatial_gate = torch.maximum(outside_rim * warmup, rim_deadlock)
            is_rough_world = (group.surface_max_height > 1e-4).to(tip.dtype)
            hole_f = rotate(q, group.raw_wrench[:, :3])
            horiz_f = torch.linalg.vector_norm(hole_f[:, :2], dim=-1)
            force_wall_gate = ((horiz_f - 0.6) / 0.8).clamp(0.0, 1.0)
            perched_gate = ((lowest_z - 0.0004 * cfg.scale) / (0.0006 * cfg.scale)).clamp(0.0, 1.0)
            step_contact_gate = torch.maximum(force_wall_gate, perched_gate)
            clear_z = group.surface_max_height + 0.0015 * cfg.scale
            below_clear = ((clear_z - lowest_z) / (0.002 * cfg.scale)).clamp(0.0, 1.0)
            gates.append(
                is_rough_world * step_contact_gate * z_weight
                * not_at_goal * stalled * spatial_gate * below_clear
            )
        return torch.cat(gates, dim=0)

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        dones = result[2]
        terminal = result[3]["terminal"]
        cfg = self.insertion_cfg
        offset = 0
        for group in self.groups:
            s = group.shape
            nacon = self.nacon_views[s].to(dtype=torch.int64, device=self.device)
            max_nefc = self.nefc_views[s].amax().to(dtype=torch.int64, device=self.device).unsqueeze(0)
            self.peak_nacon[s] = torch.maximum(self.peak_nacon[s], nacon)
            self.peak_nefc[s] = torch.maximum(self.peak_nefc[s], max_nefc)
            overflow = (nacon >= self.nacon_limits[s]) | (max_nefc >= self.njmax_limits[s])
            self.contact_overflow_steps[s].add_(overflow.to(torch.int64))

            sec = slice(offset, offset + group.n)
            pos, q, _, _ = group.state()
            tip = tip_pose(pos, q, group.offset)
            prog_pos = torch.cat((tip[:, :2], tip[:, 2:3].clamp_max(0.0)), dim=-1)
            eff_speed = torch.linalg.vector_norm(prog_pos - group.tip_ema, dim=-1) / cfg.stagnation_window_seconds
            is_stalled = (
                (eff_speed < cfg.stagnation_speed_threshold_mps * 0.7)
                & (tip[:, 2] > group.goal[:, 2] + 0.0005)
                & ~dones[sec].bool()
            )
            self.rim_stall_steps[s] = torch.where(
                is_stalled, self.rim_stall_steps[s] + 1, torch.zeros_like(self.rim_stall_steps[s]))

            ended_success = dones[sec] & (terminal["reason"][sec] == 1)
            ep_len = terminal["length"][sec]
            escaped_masks = torch.stack((
                ended_success & (ep_len > 400),
                ended_success & (ep_len > 1500),
                ended_success & (ep_len > 3000),
            ))
            self.escaped_stuck_counts[s].add_(escaped_masks.sum(dim=1).to(torch.int64))
            offset += group.n
        return result

    def contact_diagnostics_report(self) -> dict:
        lengths = self.episode_length_buf.float()
        quantiles = torch.quantile(
            lengths, torch.tensor([0.5, 0.9, 0.95, 0.99], device=lengths.device)
        ).cpu().tolist()
        shapes_report = {}
        any_overflow = False
        for s in self.peak_nacon:
            p_nacon = int(self.peak_nacon[s].item())
            p_nefc = int(self.peak_nefc[s].item())
            ov_steps = int(self.contact_overflow_steps[s].item())
            any_overflow = any_overflow or (ov_steps > 0)
            esc8, esc30, esc60 = self.escaped_stuck_counts[s].cpu().tolist()
            shapes_report[s] = {
                "peak_nacon": p_nacon,
                "naconmax": self.nacon_limits[s],
                "nacon_utilization": round(p_nacon / max(1, self.nacon_limits[s]), 6),
                "peak_nefc": p_nefc,
                "njmax": self.njmax_limits[s],
                "nefc_utilization": round(p_nefc / max(1, self.njmax_limits[s]), 6),
                "contact_overflow_steps": ov_steps,
                "escaped_success_gt_8s": int(esc8),
                "escaped_success_gt_30s": int(esc30),
                "escaped_success_gt_60s": int(esc60),
            }
        return {
            "timestamp": time.time(),
            "contact_overflow_detected": any_overflow,
            "active_episode_steps": {
                "median": round(quantiles[0], 1),
                "p90": round(quantiles[1], 1),
                "p95": round(quantiles[2], 1),
                "p99": round(quantiles[3], 1),
                "max": int(self.episode_length_buf.max().item()),
                "stuck_over_8s_envs": int((self.episode_length_buf > 400).sum().item()),
                "stuck_over_30s_envs": int((self.episode_length_buf > 1500).sum().item()),
                "stuck_over_60s_envs": int((self.episode_length_buf > 3000).sum().item()),
                "total_envs": int(self.num_envs),
            },
            "shapes": shapes_report,
        }


class RoughStagePPO(PPO):
    """Rao-Blackwellized PPO with annealed pathwise wedge-lift bootstrap and hard KL/LR guardrails."""

    rough_env: RoughLiveInsertionEnv
    wedge_gates: torch.Tensor
    escape_loss_coef: float
    escape_half_life: float
    lift_target: float
    lift_scale: float
    min_lr: float
    max_lr: float
    update_count: int

    def attach_rough_escape(self, env: RoughLiveInsertionEnv, *, escape_loss_coef: float,
                            lift_target: float, lift_scale: float,
                            min_lr: float = 1.5e-5, max_lr: float = 5.0e-5,
                            escape_half_life: float = 600.0) -> None:
        self.rough_env = env
        self.escape_loss_coef = float(escape_loss_coef)
        self.escape_half_life = max(1.0, float(escape_half_life))
        self.lift_target = float(lift_target)
        self.lift_scale = float(lift_scale)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.update_count = 0
        self.learning_rate = min(self.max_lr, max(self.min_lr, float(self.learning_rate)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate
        self.wedge_gates = torch.zeros(
            self.storage.num_transitions_per_env, self.storage.num_envs, device=self.device)

    def act(self, obs):
        if hasattr(self, "rough_env") and self.storage.step < self.wedge_gates.shape[0]:
            self.wedge_gates[self.storage.step].copy_(self.rough_env.current_wedge_gate())
        return super().act(obs)

    def update(self) -> dict[str, float]:
        if not hasattr(self, "rough_env"):
            return super().update()

        active_coef = self.escape_loss_coef * (
            0.08 + 0.92 * (0.5 ** (self.update_count / self.escape_half_life))
        )
        self.update_count += 1

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_escape_loss = 0.0
        executed_updates = 0
        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)
        mini_batch_size = self.storage.num_envs // self.num_mini_batches

        for batch_idx, batch in enumerate(generator):
            mb_i = batch_idx % self.num_mini_batches
            start = mb_i * mini_batch_size
            stop = (mb_i + 1) * mini_batch_size
            gate_batch = self.wedge_gates[:, start:stop]
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 3.5 and executed_updates > 0:
                        self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                        break
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.max_lr, self.learning_rate * 1.3)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            if active_coef > 0:
                pred_mean = distribution_params[0]
                deficit_z = (self.lift_target - pred_mean[..., 2]).clamp_min(0.0) / self.lift_scale
                escape_loss = active_coef * (
                    gate_batch * F.smooth_l1_loss(
                        deficit_z, torch.zeros_like(deficit_z), reduction="none", beta=0.15)
                ).mean()
            else:
                escape_loss = torch.zeros((), device=self.device)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + escape_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_escape_loss += escape_loss.item()
            executed_updates += 1

        denom = max(1, executed_updates)
        self.storage.clear()
        return {
            "value": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": mean_entropy / denom,
            "wedge_escape": mean_escape_loss / denom,
        }


class RoughStageRunner(ContinuousRunner):
    env: RoughLiveInsertionEnv

    def __init__(self, env: RoughLiveInsertionEnv, config: dict, output: Path, metadata: dict):
        super().__init__(env, config, output, metadata)
        escape_coef = float(metadata.get("escape_loss_coef", 0.35))
        max_lr = float(metadata.get("max_lr", 5.0e-5))
        lift_target = float(metadata.get(
            "escape_lift_target", metadata["environment"].get("stagnation_lift_target", 5.5)))
        self.alg.__class__ = RoughStagePPO
        assert isinstance(self.alg, RoughStagePPO)
        self.alg.attach_rough_escape(
            env, escape_loss_coef=escape_coef, lift_target=lift_target,
            lift_scale=10.0, min_lr=1.5e-5, max_lr=max_lr)

    def save(self, path: str, infos: dict | None = None) -> None:
        stopped = False
        try:
            super().save(path, infos)
        except TrainingStopped:
            stopped = True
        it = self.current_learning_iteration
        if it > 0 and it % 250 == 0:
            arch_dir = self.output / "checkpoints"
            arch_dir.mkdir(exist_ok=True)
            arch_target = arch_dir / f"checkpoint_{it:08d}.pt"
            if not arch_target.exists() and (self.output / "checkpoint.pt").is_file():
                tmp_arch = arch_dir / f"checkpoint_{it:08d}.pt.tmp"
                shutil.copyfile(self.output / "checkpoint.pt", tmp_arch)
                tmp_arch.replace(arch_target)
        if isinstance(self.env, RoughLiveInsertionEnv):
            atomic_json(self.output / "contact_diagnostics.json", {
                "iteration": self.current_learning_iteration,
                "policy_steps": self.env.policy_steps,
                **self.env.contact_diagnostics_report(),
            })
        if stopped:
            raise TrainingStopped()


def build_stage_metadata(source_saved: dict, args: argparse.Namespace,
                         source_path: Path) -> dict:
    source_env = dict(source_saved["infos"]["environment"])
    base_cfg = InsertionCfg.from_checkpoint(source_env)
    cfg = replace(
        base_cfg,
        shapes=tuple(args.shapes),
        envs_per_shape=args.envs_per_shape,
        shape_env_counts=(),
        adaptive_sampling=False,
        seed=args.seed,
        clearance=0.0001,
        clearance_curriculum=False,
        curriculum_steps=1,
        noise_curriculum=False,
        noise_curriculum_mode="steps",
        episode_seconds=args.episode_seconds,
        enable_timeout=args.enable_timeout,
        timeout_penalty=args.timeout_penalty if args.enable_timeout else 0.0,
        terminate_on_overload=False,
        grasp_position_error=args.grasp_position_error,
        grasp_z_error=args.grasp_z_error,
        rotation_bias=math.radians(args.rotation_bias_deg),
        hole_depth_m=args.hole_depth,
        target_depth_min_m=args.target_depth_min,
        target_depth_max_fraction=args.target_depth_max / args.hole_depth,
        align_goal_z_to_observation=True,
        workspace_z_max_m=args.workspace_z_max,
        surface_z_free_m=args.surface_z_free,
        stagnation_cost_per_second=args.stagnation_cost_per_second,
        stagnation_speed_threshold_mps=args.stagnation_speed_threshold,
        stagnation_window_seconds=args.stagnation_window_seconds,
        safety_failure_penalty=max(base_cfg.safety_failure_penalty, 400.0),
        rough_surface=True,
        rough_low_spawn_prob=args.rough_low_spawn_prob,
        rough_min_amplitude_m=args.rough_min_amplitude,
        rough_max_amplitude_m=args.rough_max_amplitude,
        rough_wavelength_min_m=args.rough_wavelength_min,
        rough_wavelength_max_m=args.rough_wavelength_max,
        rough_max_slope_deg=args.rough_max_slope_deg,
        rough_flat_probability=args.rough_flat_probability,
        rough_num_waves=args.rough_num_waves,
        rough_step_jitter=args.rough_step_jitter,
        rough_stepped_prob=args.rough_stepped_prob,
        physics_cuda_graph=args.physics_cuda_graph,
        observation_cuda_graph=args.observation_cuda_graph,
    )
    total_envs = cfg.envs_per_shape * len(cfg.shapes)
    if total_envs % 4:
        raise ValueError("total environment count must be divisible by 4")
    runner = copy.deepcopy(source_saved["infos"]["runner"])
    runner["seed"] = args.seed
    runner["save_interval"] = 1
    runner["check_for_nan"] = True
    runner["algorithm"]["learning_rate"] = args.initial_lr
    runner["algorithm"]["desired_kl"] = args.desired_kl
    runner["algorithm"]["num_mini_batches"] = 8 if total_envs % 8 == 0 else 4
    if not 0.01 <= args.min_action_std < args.max_action_std <= 1.5:
        raise ValueError("require 0.01 <= min_action_std < max_action_std <= 1.5")
    if not args.min_action_std <= args.init_action_std <= args.max_action_std:
        raise ValueError("init_action_std must lie within [min_action_std, max_action_std]")
    runner["actor"]["distribution_cfg"]["std_range"] = [args.min_action_std, args.max_action_std]
    runner["actor"]["distribution_cfg"]["init_std"] = args.init_action_std
    return {
        "stage": "rough_surface_no_timeout",
        "source_checkpoint": str(source_path.resolve()),
        "source_iteration": int(source_saved["iter"]),
        "source_policy_steps": int(source_saved["infos"]["progress"]["policy_steps"]),
        "escape_loss_coef": args.escape_loss_coef,
        "escape_lift_target": args.escape_lift_target,
        "max_lr": args.max_lr,
        "environment": asdict(cfg),
        "runner": runner,
        "contract": contract(cfg),
        "simulation_devices": [args.device],
        "collection_backend": "serial",
    }


def bootstrap_stage(output: Path, source_checkpoint: Path, args: argparse.Namespace) -> None:
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    metadata = build_stage_metadata(saved, args, source_checkpoint)
    shutil.copyfile(source_checkpoint, output / "bootstrap_source.pt")
    atomic_json(output / "config.json", metadata)
    atomic_json(output / "stage_manifest.json", {
        "created_at": time.time(),
        "source_checkpoint": metadata["source_checkpoint"],
        "source_iteration": metadata["source_iteration"],
        "source_policy_steps": metadata["source_policy_steps"],
        "rough_surface": metadata["contract"]["rough_surface"],
        "std_range": metadata["runner"]["actor"]["distribution_cfg"]["std_range"],
        "desired_kl": metadata["runner"]["algorithm"]["desired_kl"],
        "initial_lr": metadata["runner"]["algorithm"]["learning_rate"],
        "max_lr": metadata["max_lr"],
        "escape_loss_coef": metadata["escape_loss_coef"],
        "escape_lift_target": metadata["escape_lift_target"],
    })
    logging.info(
        "Bootstrapped rough-surface stage at %s from %s (iter %d).",
        output, source_checkpoint, metadata["source_iteration"],
    )


def rough_worker(output: Path, device: str, attempt: int) -> None:
    metadata = json.loads((output / "config.json").read_text())
    cfg = InsertionCfg.from_checkpoint(metadata["environment"])
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed + attempt)
    env = RoughLiveInsertionEnv(cfg, device, output, metadata)
    runner = RoughStageRunner(env, metadata["runner"], output, metadata)

    if (output / "checkpoint.pt").exists():
        resume_training(runner, cfg, attempt)
        max_lr = float(metadata.get("max_lr", 5.0e-5))
        runner.alg.learning_rate = min(max_lr, max(1.5e-5, float(runner.alg.learning_rate)))
        for pg in runner.alg.optimizer.param_groups:
            pg["lr"] = runner.alg.learning_rate
        if isinstance(runner.alg, RoughStagePPO):
            runner.alg.update_count = int(runner.current_learning_iteration)
    else:
        bootstrap_path = output / "bootstrap_source.pt"
        if not bootstrap_path.is_file():
            raise FileNotFoundError(f"missing bootstrap_source.pt in {output}")
        source_saved = torch.load(bootstrap_path, map_location=device, weights_only=True)
        runner.alg.load(source_saved, None, True)
        max_lr = float(metadata.get("max_lr", 5.0e-5))
        target_lr = min(max_lr, max(1.5e-5, float(metadata["runner"]["algorithm"]["learning_rate"])))
        runner.alg.learning_rate = target_lr
        for pg in runner.alg.optimizer.param_groups:
            pg["lr"] = target_lr
        old_std_range = source_saved["infos"]["runner"]["actor"]["distribution_cfg"]["std_range"]
        new_std_range = metadata["runner"]["actor"]["distribution_cfg"]["std_range"]
        init_std = float(metadata["runner"]["actor"]["distribution_cfg"].get("init_std", 0.40))
        if float(old_std_range[1]) <= 0.25 and tuple(old_std_range) != tuple(new_std_range):
            head = runner.alg._raw_actor.mlp[-2]
            with torch.no_grad():
                head.weight[6:].zero_()
                head.bias[6:].fill_(math.log(init_std))
                for p in (head.weight, head.bias):
                    st = runner.alg.optimizer.state.get(p)
                    if st:
                        if "exp_avg" in st:
                            st["exp_avg"][6:].zero_()
                        if "exp_avg_sq" in st:
                            st["exp_avg_sq"][6:].zero_()
            logging.info(
                "Reset narrow actor log_std head to init_std=%.3f within std_range %s -> %s (lr=%.2e).",
                init_std, old_std_range, new_std_range, target_lr,
            )
        else:
            logging.info(
                "Preserved learned actor log_std head from %s within std_range %s (lr=%.2e).",
                old_std_range, new_std_range, target_lr,
            )
        runner.current_learning_iteration = 0
        progress = source_saved["infos"].get("progress", {})
        env.policy_steps = int(progress.get("policy_steps", 0))
        env.total_transitions = int(progress.get("transitions", 0))
        logging.info(
            "Warm-started actor, critic, normalizers and Adam preconditioner from %s "
            "(source iter %d); starting Stage-2 iteration 0.",
            bootstrap_path, int(source_saved["iter"]),
        )

    # 第二阶段全量开启难度与噪声，使首次 rollout 即处于完整凹凸表面分布
    for index, group in enumerate(env.groups):
        group.difficulty = 1.0
        group.noise_scale = 1.0
        seed = cfg.seed + index * 10007 + env.policy_steps + attempt
        group.generator.manual_seed(seed)
        group.walk_generator.manual_seed(seed + 500003)
        group.reset(torch.arange(group.n, device=group.device))

    def request_stop(signum, frame):
        del frame
        logging.info("Stop signal %d: saving after the current PPO iteration.", signum)
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        runner.learn(1_000_000)
    except TrainingStopped:
        if runner.logger.writer is not None:
            runner.logger.stop_logging_writer()
        logging.info("Rough-surface worker stopped cleanly with atomic checkpoint.")


def evaluate_stage_health(output: Path, state: str, heartbeat: dict | None) -> dict:
    diag_path = output / "contact_diagnostics.json"
    term_path = output / "termination_counts.json"
    diag = json.loads(diag_path.read_text()) if diag_path.is_file() else None
    term = json.loads(term_path.read_text()) if term_path.is_file() else None
    issues: list[str] = []
    if state != "running":
        issues.append(f"worker_state={state}")
    if heartbeat is None:
        issues.append("missing_heartbeat")
    elif heartbeat.get("mean_episode_reward") is not None and not math.isfinite(
            heartbeat["mean_episode_reward"]):
        issues.append("nonfinite_mean_reward")
    if diag is not None and diag.get("contact_overflow_detected", False):
        issues.append("contact_or_constraint_buffer_overflow")
    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "iteration": None if heartbeat is None else heartbeat.get("iteration"),
        "mean_episode_reward": None if heartbeat is None else heartbeat.get("mean_episode_reward"),
        "contact_diagnostics": diag,
        "termination_summary": None if term is None else term.get("shapes"),
    }


@torch.inference_mode()
def validate_checkpoint_cli(output: Path, ckpt_path: Path, device: str, report_out: Path) -> None:
    from evaluate import load_checkpoint
    from insertion import InsertionEnv

    torch.set_num_threads(1)
    saved = torch.load(ckpt_path, map_location=device, weights_only=True)
    meta_cfg = json.loads((output / "config.json").read_text())
    base_cfg = InsertionCfg.from_checkpoint(meta_cfg["environment"])
    _, actor = load_checkpoint(ckpt_path, device)

    def run_suite(is_rough: bool, max_steps: int) -> dict:
        cfg_eval = replace(
            base_cfg,
            envs_per_shape=512,
            shape_env_counts=(),
            training=False,
            rough_surface=is_rough,
            rough_low_spawn_prob=0.5 if is_rough else 0.0,
            rough_min_amplitude_m=base_cfg.rough_min_amplitude_m if is_rough else 0.0,
            rough_max_amplitude_m=base_cfg.rough_max_amplitude_m if is_rough else 0.0,
            rough_flat_probability=0.0,
            episode_seconds=max_steps * 0.02,
            enable_timeout=True,
            physics_cuda_graph=True,
            observation_cuda_graph=True,
        )
        env = InsertionEnv(cfg_eval, device)
        for idx, g in enumerate(env.groups):
            g.difficulty = 1.0
            g.noise_scale = 1.0
            g.generator.manual_seed(90001 + idx * 1009)
            g.walk_generator.manual_seed(190001 + idx * 1009)
            g.reset(torch.arange(g.n, device=device))
        obs = env.get_observations()
        actor.reset()
        n = env.num_envs
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        succ_8s = torch.zeros(n, dtype=torch.bool, device=device)
        succ_end = torch.zeros(n, dtype=torch.bool, device=device)
        lift_ever = torch.zeros(n, dtype=torch.bool, device=device)
        finish_step = torch.zeros(n, dtype=torch.long, device=device)
        for step in range(max_steps):
            act = actor(obs, stochastic_output=False)
            if step >= 50:
                lift_ever |= (~finished) & (act[:, 2] > 1.0)
            obs, _, dones, info = env.step(act)
            actor.reset(dones)
            newly = dones.bool() & (~finished)
            is_succ = newly & (info["terminal"]["reason"] == 1)
            if step < 400:
                succ_8s |= is_succ
            succ_end |= is_succ
            finish_step = torch.where(is_succ, torch.full_like(finish_step, step + 1), finish_step)
            finished |= newly
        shapes_sr = {}
        off = 0
        for g in env.groups:
            sl = slice(off, off + g.n)
            shapes_sr[g.shape] = round(float(succ_end[sl].float().mean().item()), 4)
            off += g.n
        mean_t = round(float(finish_step[succ_end].float().mean().item() * 0.02), 3) if succ_end.any() else 0.0
        return {
            "sr_8s": round(float(succ_8s.float().mean().item()), 4),
            "sr_end": round(float(succ_end.float().mean().item()), 4),
            "mean_success_seconds": mean_t,
            "lift_episodes": int(lift_ever.sum().item()),
            "lift_assisted_successes": int((lift_ever & succ_end).sum().item()),
            "total_episodes": int(n),
            "per_shape": shapes_sr,
        }

    flat_res = run_suite(False, 400)
    rough_res = run_suite(True, 750)
    atomic_json(report_out, {
        "evaluated_at": time.time(),
        "checkpoint": str(ckpt_path),
        "iteration": int(saved["iter"]),
        "flat_8s": flat_res,
        "rough_15s": rough_res,
    })


def run_isolated_validation(output: Path, ckpt_source: Path, device: str) -> dict | None:
    if not ckpt_source.is_file():
        return None
    snap = output / "eval_snapshot.pt"
    report_out = output / "eval_latest.json"
    shutil.copyfile(ckpt_source, snap)
    res = subprocess.run([
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--validate-checkpoint", str(snap),
        "--validate-report", str(report_out),
        "--output", str(output),
        "--device", device,
    ], check=False, timeout=900)
    if res.returncode != 0 or not report_out.is_file():
        logging.warning("Validation subprocess exited with code %s", res.returncode)
        return None
    return json.loads(report_out.read_text())


def supervise_rough_stage(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not args.resume:
        bootstrap_stage(output, args.source_checkpoint, args)
    elif not (output / "config.json").is_file():
        raise FileNotFoundError(f"resume requires {output / 'config.json'}")

    if (output / "STOP").exists():
        raise RuntimeError("STOP exists; remove STOP before resuming")

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        del frame
        stopping = True
        logging.info("Supervisor received stop signal %d.", signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    best_ckpt = output / "best_checkpoint.pt"
    guard_state_path = output / "guard_state.json"
    if guard_state_path.is_file():
        guard = json.loads(guard_state_path.read_text())
    else:
        seed_ckpt = output / ("checkpoint.pt" if (output / "checkpoint.pt").is_file() else "bootstrap_source.pt")
        shutil.copyfile(seed_ckpt, best_ckpt)
        logging.info("Running baseline paired validation (N=2560 Flat + N=2560 Fine-Rough) on %s ...", seed_ckpt)
        base_eval = run_isolated_validation(output, best_ckpt, args.device)
        if base_eval is None:
            raise RuntimeError("Initial baseline validation failed")
        b_flat = float(base_eval["flat_8s"]["sr_end"])
        b_rough = float(base_eval["rough_15s"]["sr_end"])
        guard = {
            "baseline_flat_sr": b_flat,
            "baseline_rough_sr": b_rough,
            "best_flat_sr": b_flat,
            "best_rough_sr": b_rough,
            "best_score": round(0.65 * b_rough + 0.35 * b_flat, 5),
            "best_iteration": int(base_eval["iteration"]),
            "consecutive_degraded": 0,
            "rollbacks": 0,
        }
        atomic_json(guard_state_path, guard)
        with (output / "hourly_evaluations.jsonl").open("a", buffering=1) as eval_log:
            eval_log.write(json.dumps({
                "phase": "baseline_seed",
                "promoted_best": True,
                "guard": guard,
                **base_eval,
            }, allow_nan=False) + "\n")
        logging.info("Seeded best_checkpoint.pt with baseline: flat=%.2f%%, rough@15s=%.2f%%.",
                     b_flat * 100, b_rough * 100)

    stage_start = time.time()

    def current_interval(now: float) -> float:
        if now - stage_start < args.initial_phase_seconds:
            return args.initial_check_seconds
        return args.steady_check_seconds

    next_check = stage_start + args.initial_check_seconds
    attempt = 0
    process = None
    with (
        (output / "supervisor.jsonl").open("a", buffering=1) as events,
        (output / "health_checks.jsonl").open("a", buffering=1) as health_log,
        (output / "hourly_evaluations.jsonl").open("a", buffering=1) as eval_log,
    ):
        def record(kind: str, **details) -> None:
            event = {"timestamp": time.time(), "kind": kind, **details}
            events.write(json.dumps(event, allow_nan=False) + "\n")
            logging.info("%s", event)

        try:
            while not stopping and not (output / "STOP").exists():
                with (output / "training.log").open("a", buffering=1) as log:
                    started = time.time()
                    process = subprocess.Popen([
                        sys.executable, "-u", str(Path(__file__).resolve()),
                        "--worker", "--output", str(output),
                        "--device", args.device, "--attempt", str(attempt),
                    ], stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(output / "processes.json", {
                        "supervisor_pid": os.getpid(),
                        "worker_pid": process.pid,
                        "attempt": attempt,
                        "started": started,
                    })
                    record("started", pid=process.pid, attempt=attempt)
                    rollback_triggered = False
                    while not stopping and not (output / "STOP").exists():
                        hb_path = output / "heartbeat.json"
                        heartbeat = json.loads(hb_path.read_text()) if hb_path.exists() else None
                        now = time.time()
                        state = health_state(process.poll(), now, started, heartbeat, args.stall_seconds)
                        if now >= next_check:
                            phase = ("initial_check"
                                     if now - stage_start < args.initial_phase_seconds
                                     else "steady_hourly")
                            report = evaluate_stage_health(output, state, heartbeat)
                            val_res = None
                            promoted = False
                            if (output / "checkpoint.pt").is_file():
                                val_res = run_isolated_validation(output, output / "checkpoint.pt", args.device)
                            if val_res is not None:
                                f_sr = float(val_res["flat_8s"]["sr_end"])
                                r_sr = float(val_res["rough_15s"]["sr_end"])
                                score = round(0.65 * r_sr + 0.35 * f_sr, 5)
                                flat_safe = f_sr >= guard["baseline_flat_sr"] - 0.012
                                if flat_safe and score > guard["best_score"]:
                                    tmp_best = output / "best_checkpoint.pt.tmp"
                                    shutil.copyfile(output / "eval_snapshot.pt", tmp_best)
                                    tmp_best.replace(best_ckpt)
                                    guard["best_flat_sr"] = f_sr
                                    guard["best_rough_sr"] = r_sr
                                    guard["best_score"] = score
                                    guard["best_iteration"] = int(val_res["iteration"])
                                    guard["consecutive_degraded"] = 0
                                    promoted = True
                                elif (f_sr < guard["baseline_flat_sr"] - 0.030
                                      or r_sr < guard["best_rough_sr"] - 0.050):
                                    guard["consecutive_degraded"] = int(guard["consecutive_degraded"]) + 1
                                else:
                                    guard["consecutive_degraded"] = 0
                                atomic_json(guard_state_path, guard)
                                eval_entry = {
                                    "phase": phase,
                                    "promoted_best": promoted,
                                    "score": score,
                                    "guard": copy.deepcopy(guard),
                                    **val_res,
                                }
                                eval_log.write(json.dumps(eval_entry, allow_nan=False) + "\n")
                                record("scheduled_evaluation", phase=phase,
                                       iteration=val_res["iteration"],
                                       flat_sr=f_sr, rough_sr_15s=r_sr,
                                       rough_sr_8s=val_res["rough_15s"]["sr_8s"],
                                       promoted_best=promoted,
                                       consecutive_degraded=guard["consecutive_degraded"])
                            check_entry = {
                                "timestamp": time.time(),
                                "elapsed_seconds": round(time.time() - stage_start, 1),
                                "phase": phase,
                                "pid": process.pid,
                                "validation": None if val_res is None else {
                                    "flat_sr": val_res["flat_8s"]["sr_end"],
                                    "rough_sr_8s": val_res["rough_15s"]["sr_8s"],
                                    "rough_sr_15s": val_res["rough_15s"]["sr_end"],
                                    "promoted_best": promoted,
                                },
                                **report,
                            }
                            health_log.write(json.dumps(check_entry, allow_nan=False) + "\n")
                            record("scheduled_health_check", phase=phase,
                                   healthy=report["healthy"], issues=report["issues"],
                                   iteration=report["iteration"])
                            next_check = time.time() + current_interval(time.time())
                            if int(guard["consecutive_degraded"]) >= 2 and best_ckpt.is_file():
                                record("auto_rollback_to_best",
                                       best_iteration=guard["best_iteration"],
                                       best_flat_sr=guard["best_flat_sr"],
                                       best_rough_sr=guard["best_rough_sr"])
                                stop_worker(process, args.stop_grace_seconds)
                                tmp_restore = output / "checkpoint.pt.tmp"
                                shutil.copyfile(best_ckpt, tmp_restore)
                                tmp_restore.replace(output / "checkpoint.pt")
                                meta_now = json.loads((output / "config.json").read_text())
                                new_max_lr = max(2.0e-5, float(meta_now.get("max_lr", 5.0e-5)) * 0.75)
                                meta_now["max_lr"] = new_max_lr
                                meta_now["escape_loss_coef"] = max(
                                    0.05, float(meta_now.get("escape_loss_coef", 0.35)) * 0.70)
                                atomic_json(output / "config.json", meta_now)
                                guard["consecutive_degraded"] = 0
                                guard["rollbacks"] = int(guard["rollbacks"]) + 1
                                atomic_json(guard_state_path, guard)
                                rollback_triggered = True
                                break
                            # Re-read heartbeat after validation subprocess so stall check isn't tripped by eval runtime
                            started = max(started, time.time() - 60)
                            continue
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
                delay_s = 5 if rollback_triggered else 30
                record("restarting", attempt=attempt, delay_seconds=delay_s)
                for _ in range(delay_s):
                    if stopping or (output / "STOP").exists():
                        break
                    time.sleep(1)
        finally:
            if process is not None:
                stop_worker(process, args.stop_grace_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/rough_surface_no_timeout"))
    parser.add_argument("--source-checkpoint", type=Path,
                        default=Path("runs/continuous_position_delay20mm/checkpoint.pt"),
                        help="source checkpoint for warm-starting actor/critic/normalizers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs-per-shape", type=int, default=512)
    parser.add_argument("--shapes", nargs="+", choices=TRAIN_SHAPES, default=list(TRAIN_SHAPES))
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--initial-lr", type=float, default=3.5e-5)
    parser.add_argument("--max-lr", type=float, default=5.0e-5)
    parser.add_argument("--desired-kl", type=float, default=0.010)
    parser.add_argument("--rough-min-amplitude", type=float, default=0.0012,
                        help="minimum amplitude on rough worlds in meters (default 1.2 mm)")
    parser.add_argument("--rough-max-amplitude", type=float, default=0.004,
                        help="maximum peak-to-valley surface roughness in meters (default 4.0 mm)")
    parser.add_argument("--rough-step-jitter", type=float, default=0.30,
                        help="per-brick random height fragmentation fraction")
    parser.add_argument("--rough-stepped-prob", type=float, default=0.50,
                        help="fraction of rough worlds using flat-topped 90-deg stepped micro-bricks")
    parser.add_argument("--rough-low-spawn-prob", type=float, default=0.5,
                        help="fraction of rough-surface resets spawned right above local valley terrain")
    parser.add_argument("--workspace-z-max", type=float, default=0.06,
                        help="upper Z workspace escape ceiling in meters for Stage 2")
    parser.add_argument("--surface-z-free", type=float, default=0.006,
                        help="height-free obstacle-crossing Z band [0, h] above hole plane in meters")
    parser.add_argument("--min-action-std", type=float, default=0.06,
                        help="lower bound on policy action exploration std in Stage 2 to avoid KL lock")
    parser.add_argument("--max-action-std", type=float, default=0.65,
                        help="upper bound on policy action exploration std in Stage 2")
    parser.add_argument("--init-action-std", type=float, default=0.40,
                        help="unclamped warm-start exploration std when migrating from narrow std_range")
    parser.add_argument("--episode-seconds", type=float, default=30.0,
                        help="long-horizon anti-zombie episode budget in seconds for rough-surface training")
    parser.add_argument("--enable-timeout", action=argparse.BooleanOptionalAction, default=True,
                        help="recycle deadlocked rough-surface episodes after --episode-seconds")
    parser.add_argument("--timeout-penalty", type=float, default=60.0,
                        help="terminal penalty when a rough-surface episode hits the anti-zombie deadline")
    parser.add_argument("--grasp-position-error", type=float, default=0.0015,
                        help="per-axis Gaussian std of lateral XY TCP/grasp offset in meters")
    parser.add_argument("--grasp-z-error", type=float, default=0.005,
                        help="Gaussian std of longitudinal Z TCP/grasp offset in meters")
    parser.add_argument("--hole-depth", type=float, default=0.028)
    parser.add_argument("--target-depth-min", type=float, default=0.003)
    parser.add_argument("--target-depth-max", type=float, default=0.025)
    parser.add_argument("--rotation-bias-deg", type=float, default=1.5,
                        help="per-axis uniform half range of constant orientation bias in degrees")
    parser.add_argument("--stagnation-cost-per-second", type=float, default=2.5,
                        help="running penalty per second when pre-goal smoothed tip speed stagnates")
    parser.add_argument("--escape-loss-coef", type=float, default=0.35,
                        help="annealed Rao-Blackwellized pathwise lift bootstrap weight on wedged states")
    parser.add_argument("--escape-lift-target", type=float, default=5.5,
                        help="target upward Z action (in mm units) when wedged below surface clearance")
    parser.add_argument("--stagnation-window-seconds", type=float, default=0.7,
                        help="EMA window in seconds for net tip displacement stagnation detection")
    parser.add_argument("--stagnation-speed-threshold", type=float, default=0.0025,
                        help="net displacement speed threshold in m/s below which stagnation penalty applies")
    parser.add_argument("--rough-wavelength-min", type=float, default=0.005,
                        help="minimum spatial wavelength in meters (default 5 mm)")
    parser.add_argument("--rough-wavelength-max", type=float, default=0.028,
                        help="maximum spatial wavelength in meters (default 28 mm)")
    parser.add_argument("--rough-max-slope-deg", type=float, default=16.0)
    parser.add_argument("--rough-flat-probability", type=float, default=0.25)
    parser.add_argument("--rough-num-waves", type=int, default=5)
    parser.add_argument("--physics-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-check-seconds", type=float, default=900.0,
                        help="health & eval interval during initial phase (default 900s = 15 min)")
    parser.add_argument("--initial-phase-seconds", type=float, default=1800.0,
                        help="duration of initial check phase before hourly cadence (default 1800s)")
    parser.add_argument("--steady-check-seconds", type=float, default=3600.0,
                        help="hourly supervision & dual-benchmark evaluation cadence (default 3600s = 1 hour)")
    parser.add_argument("--stall-seconds", type=float, default=900.0)
    parser.add_argument("--stop-grace-seconds", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--validate-checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-report", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.validate_checkpoint is not None:
        assert args.validate_report is not None
        validate_checkpoint_cli(
            args.output.resolve(), args.validate_checkpoint.resolve(),
            args.device, args.validate_report.resolve())
    elif args.worker:
        rough_worker(args.output.resolve(), args.device, args.attempt)
    else:
        supervise_rough_stage(args)


if __name__ == "__main__":
    main()
