"""Leave-one-error-out evaluation of frozen ordinary control."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel

from evaluate import load_checkpoint, REASONS
from insertion import InsertionCfg, InsertionEnv
from insertion.transforms import rotate, tip_pose


ERRORS = (
    "position_bias", "position_noise", "rotation_noise", "tip_calibration_error",
    "grasp_position_error", "sensor_rotation_error", "force_noise", "moment_noise",
    "force_bias", "moment_bias", "delay",
)

GRASP_CHANNELS = (
    "none", "grasp_position_observation", "grasp_wrench_observation",
    "grasp_both_observations", "grasp_position_error",
)


def compensate_grasp_observations(group, position: bool, wrench: bool) -> None:
    original = group.observations_from_noise

    def observations(unused, noise, position_noise, rotation_noise):
        actor, critic = original(noise, position_noise, rotation_noise)
        actor = actor.clone()
        grasp = group.offset-group.nominal_offset
        if position:
            _, q, _, _ = group.state()
            actor[:, :3] -= rotate(q, grasp)
        if wrench:
            # Transport the filtered signal before independently added measurement noise/bias.
            force = rotate(group.sensor_error, group.filtered_sensor[:, :3])
            arm = rotate(group.sensor_error, grasp)
            actor[:, 10:13] -= torch.linalg.cross(arm, force)
        return actor, critic

    group.observations_from_noise = types.MethodType(observations, group)


def exact_xy_observations(group) -> None:
    original = group.observations_from_noise

    def observations(unused, noise, position_noise, rotation_noise):
        actor, critic = original(noise, position_noise, rotation_noise)
        actor = actor.clone()
        pos, q, _, _ = group.state()
        true_tip = tip_pose(pos, q, group.offset)
        actor[:, :2] = (group.goal-true_tip)[:, :2]
        return actor, critic

    group.observations_from_noise = types.MethodType(observations, group)


def exact_wrench_reference_observations(group) -> None:
    original = group.observations_from_noise

    def observations(unused, noise, position_noise, rotation_noise):
        actor, critic = original(noise, position_noise, rotation_noise)
        actor = actor.clone()
        displacement = group.offset-(group.nominal_offset+group.calibration)
        arm = rotate(group.sensor_error, displacement)
        force = rotate(group.sensor_error, group.filtered_sensor[:, :3])
        actor[:, 10:13] -= torch.linalg.cross(arm, force)
        return actor, critic

    group.observations_from_noise = types.MethodType(observations, group)


def remove_error(group, name: str) -> None:
    if name == "exact_xy_and_wrench":
        remove_error(group, "exact_wrench_no_additive_noise")
        exact_xy_observations(group)
    elif name == "exact_wrench_reference":
        exact_wrench_reference_observations(group)
    elif name == "exact_xy":
        exact_xy_observations(group)
    elif name in GRASP_CHANNELS[1:4]:
        compensate_grasp_observations(
            group, position=name != "grasp_wrench_observation",
            wrench=name != "grasp_position_observation")
    elif name in ("position_noise", "rotation_noise", "force_noise", "moment_noise"):
        group.cfg = replace(group.cfg, **{name: 0.})
    elif name == "position_bias":
        group.bias.zero_()
    elif name == "tip_calibration_error":
        group.calibration.zero_()
    elif name == "sensor_rotation_error":
        group.sensor_error.copy_(group.identity)
    elif name in ("force_bias", "moment_bias"):
        axes = slice(0, 3) if name == "force_bias" else slice(3, 6)
        group.wrench_bias[:, axes] = 0
    elif name == "exact_wrench_no_additive_noise":
        group.cfg = replace(group.cfg, force_noise=0., moment_noise=0.)
        group.wrench_bias.zero_()
        exact_wrench_reference_observations(group)
    elif name == "delay":
        group.delay.zero_()
    elif name == "grasp_position_error":
        pos, q, _, _ = group.state()
        true_tip = tip_pose(pos, q, group.offset).clone()
        group.offset.copy_(group.nominal_offset)
        group.sim.data.qpos[:, :3] = true_tip-rotate(q, group.offset)
        for geom_id in group.peg_geom_ids:
            group.sim.model.geom_pos[:, geom_id] = group.default_geom_pos[geom_id]
        group.sim.recompute_constants(RecomputeLevel.set_const)
        group.sim.forward()
    else:
        raise ValueError(f"unknown error {name}")
    pos, q, _, _ = group.state()
    group.targets_p[:] = tip_pose(pos, q, group.nominal_offset+group.calibration).unsqueeze(0)
    group.targets_q[:] = q.unsqueeze(0)


def preserve_reset_rng(group):
    original = group.reset

    def reset(unused, ids):
        physical = group.generator.get_state()
        walk = group.walk_generator.get_state()
        original(ids)
        group.generator.set_state(physical)
        group.walk_generator.set_state(walk)

    group.reset = types.MethodType(reset, group)


@torch.inference_mode()
def run(checkpoint: Path, output: Path, count: int,
        grasp_channels: bool = False, exact_xy: bool = False, exact_wrench: bool = False,
        no_wrench_noise: bool = False, combined: bool = False):
    torch.set_num_threads(1)
    metadata, actor = load_checkpoint(checkpoint, "cuda:0")
    base = InsertionCfg.from_checkpoint(metadata["environment"])
    output.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    if combined:
        conditions = ("none", "exact_xy", "exact_wrench_no_additive_noise", "exact_xy_and_wrench")
    elif no_wrench_noise:
        conditions = ("none", "exact_wrench_no_additive_noise")
    elif exact_wrench:
        conditions = ("none", "exact_wrench_reference")
    elif exact_xy:
        conditions = ("none", "exact_xy")
    elif grasp_channels:
        conditions = GRASP_CHANNELS
    else:
        conditions = ("none", *ERRORS)
    for seed in (42, 2026):
        reference = None
        for removed in conditions:
            cfg = replace(base, seed=seed, shape_env_counts=(), envs_per_shape=count,
                          adaptive_sampling=False, training=False, observation_cuda_graph=False)
            env = InsertionEnv(cfg, "cuda:0")
            states = []
            for index, group in enumerate(env.groups):
                group.difficulty = 1.
                group.generator.manual_seed(seed+index*10007)
                group.walk_generator.manual_seed(seed+index*10007+500003)
                group.reset(torch.arange(group.n, device=group.device))
                states.append(torch.cat([
                    group.sim.data.qpos[:].flatten(), group.offset.flatten(),
                    group.goal.flatten(), group.bias.flatten(), group.calibration.flatten(),
                    group.k.flatten(), group.d.flatten(), group.wrench_bias.flatten(),
                    group.sensor_error.flatten(), group.delay.float().flatten(),
                ]).cpu())
                if removed != "none":
                    remove_error(group, removed)
                preserve_reset_rng(group)
            initial = torch.cat(states)
            if reference is None:
                reference = initial
            else:
                torch.testing.assert_close(initial, reference, rtol=0, atol=0)
            hashes[f"{seed}/{removed}"] = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
            actor.reset()
            obs = env.get_observations()
            finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            shapes = [g.shape for g in env.groups for _ in range(g.n)]
            for _ in range(env.max_episode_length):
                obs, _, done, extras = env.step(actor(obs))
                for slot in (done & ~finished).nonzero().flatten().tolist():
                    terminal = extras["terminal"]
                    records.append({
                        "removed": removed, "seed": seed, "slot": slot, "shape": shapes[slot],
                        "success": int(terminal["reason"][slot]) == 1,
                        "reason": REASONS[int(terminal["reason"][slot])],
                        "seconds": float(terminal["length"][slot])*0.02,
                    })
                finished |= done
                actor.reset(done)
                if bool(finished.all()):
                    break
            if not bool(finished.all()):
                raise RuntimeError("evaluation exceeded first-episode budget")
            subset = [r for r in records if r["seed"] == seed and r["removed"] == removed]
            print(seed, removed, sum(r["success"] for r in subset), "/", len(subset), flush=True)
            (output/"episodes.json").write_text(json.dumps(records, indent=2)+"\n")
            for group in env.groups:
                del group.reset
                if removed in GRASP_CHANNELS[1:4] or removed in (
                        "exact_xy", "exact_wrench_reference", "exact_wrench_no_additive_noise",
                        "exact_xy_and_wrench"):
                    del group.observations_from_noise
            del env, group
            gc.collect()
            torch.cuda.empty_cache()
    reference = {(r["seed"], r["slot"]): r["success"]
                 for r in records if r["removed"] == "none"}
    rng = np.random.default_rng(17)
    summary = []
    for removed in conditions:
        subset = [r for r in records if r["removed"] == removed]
        differences = np.array([int(r["success"])-int(reference[r["seed"], r["slot"]])
                                for r in subset])
        summary.append({
            "removed": removed, "trials": len(subset),
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "delta": float(differences.mean()),
            "paired_bootstrap_95": np.quantile(
                differences[rng.integers(len(subset), size=(4000, len(subset)))].mean(1),
                [0.025, 0.975]).tolist(),
            "rescued": int((differences == 1).sum()), "regressed": int((differences == -1).sum()),
            "overloads": sum(r["reason"] == "wrench_limit" for r in subset),
            "by_shape": {s: float(np.mean([r["success"] for r in subset if r["shape"] == s]))
                         for s in base.shapes},
        })
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "control": "ordinary policy, original random initial orientations, full noise",
        "summary": sorted(summary, key=lambda r: r["delta"], reverse=True),
        "grasp_channel_test": grasp_channels,
        "combined_factorial_test": combined,
        "combined_definition": (
            "Four conditions: original, exact actor XY only, actual-tip wrench reference "
            "with no force/moment additive noise or bias, and both. Other channels unchanged."
            if combined else None),
        "exact_xy_test": exact_xy,
        "exact_wrench_reference_test": exact_wrench,
        "no_wrench_additive_noise": no_wrench_noise,
        "exact_wrench_reference_definition": (
            "Transport filtered wrench from estimated tip to actual tip using displacement "
            "offset - nominal_offset - calibration. Preserve sensor rotation and filter. "
            "Force/moment white noise and bias disabled only when no_wrench_additive_noise=true. "
            "All position inputs, control and safety unchanged."
            if exact_wrench else None),
        "exact_xy_definition": (
            "Actor dx/dy replaced with true goal minus actual tip XY, without target bias, "
            "calibration, grasp-reference error or position white noise in these two entries. "
            "All other actor entries, control, physical randomization and safety unchanged."
            if exact_xy else None),
        "channel_definition": (
            "Position adds actual grasp displacement to the estimated tip, retaining calibration, "
            "target bias and white noise. Wrench transports the filtered signal by actual grasp "
            "displacement, retaining calibration, sensor rotation and independent noise/bias. "
            "Neither modifies controller reference, collision geometry, reward or safety signal."
            if grasp_channels else None),
        "pre_intervention_initial_hashes": hashes,
        "note": "One error removed after identical reset draws; other error draws unchanged. "
                "Auto-resets restore generator states to avoid perturbing remaining first episodes. "
                "Grasp removal preserves actual tip pose, updates geometry and mass constants. "
                "Intervals are exploratory, not multiple-comparison adjusted; "
                "removal effects are conditional on all other errors being present.",
    }
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    channel = parser.add_mutually_exclusive_group()
    channel.add_argument("--grasp-channels", action="store_true",
                         help="split grasp compensation into position/wrench observation channels")
    channel.add_argument("--exact-xy", action="store_true",
                         help="compare full noise against exact actor dx/dy only")
    channel.add_argument("--exact-wrench-reference", action="store_true",
                         help="use actual tip wrench reference, preserving all measurement noise")
    parser.add_argument("--no-wrench-noise", action="store_true",
                        help="with exact wrench reference, disable force/moment white noise and bias")
    channel.add_argument("--combined", action="store_true",
                         help="four-condition exact XY and improved wrench factorial test")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    if args.no_wrench_noise and not args.exact_wrench_reference:
        parser.error("--no-wrench-noise requires --exact-wrench-reference")
    run(args.checkpoint, args.output, args.count,
        args.grasp_channels, args.exact_xy, args.exact_wrench_reference, args.no_wrench_noise,
        args.combined)
