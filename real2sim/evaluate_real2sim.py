"""实机档评估：固定实测对象参数并启用部署低通，统计成功率、用时、贴板停留与动作平滑度

用法（仓库根目录）:
    python -m real2sim.evaluate_real2sim baseline=runs/releases/stage2-outer-xy-best-961/checkpoint.pt \
        new=runs/real2sim/stage3_from_961/checkpoint.pt --output runs/real2sim/eval_real
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from evaluate import MapPolicy, load_checkpoint
from insertion import InsertionCfg, InsertionEnv, TRAIN_SHAPES
from insertion.env import CONTACT_THRESHOLD_N
from insertion.transforms import tip_pose

# 实测：650 N/m、4.8 N·m/rad、20 ms 延迟、Z 死区 2.5 N、下游 0.6 kg 由 dynamics="real" 固定
REAL_PROFILE = dict(
    dynamics="real", training=False, rough_surface=False, clearance=0.0001,
    initial_xy=0.018, initial_z_min=0.030, initial_z_max=0.055, workspace_z_max_m=0.07,
    target_depth_min_m=0.010, fixed_target_depth_m=None,
    target_lowpass=True, target_lowpass_translation_rad_s=30.0, target_lowpass_rotation_rad_s=50.0,
    step_bias_rate_per_second=0.0, clearance_curriculum=False, adaptive_sampling=False,
    shape_env_counts=(), vibration_force_min_n=1.0, vibration_force_max_n=1.0,
    sensor_tip_min_m=0.20, sensor_tip_max_m=0.20, target_lowpass_off_probability=0.0,
)
ACTION_THRESHOLD = 0.1
VELOCITY_THRESHOLD_M = 0.0001


def sign_flips(previous: torch.Tensor, current: torch.Tensor,
               threshold: float = ACTION_THRESHOLD) -> tuple[torch.Tensor, torch.Tensor]:
    # 仅统计相邻两步都超过阈值的分量，返回每行的反号数与有效对数
    valid = (previous.abs() > threshold) & (current.abs() > threshold)
    flips = valid & (previous.sign() != current.sign())
    return flips.sum(-1), valid.sum(-1)


def velocity_reversals(previous: torch.Tensor, current: torch.Tensor,
                       threshold: float = VELOCITY_THRESHOLD_M) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ((torch.linalg.vector_norm(previous, dim=-1) > threshold)
             & (torch.linalg.vector_norm(current, dim=-1) > threshold))
    return valid & ((previous * current).sum(-1) < 0), valid


def probe(env: InsertionEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 接触由上一控制周期内的真实约束力判断，不受惯性力与振动噪声影响
    tips, contacts, surface = [], [], []
    for group in env.groups:
        pos, q, _, _ = group.state()
        tip = tip_pose(pos, q, group.offset)
        contact = group.step_peak_contact > CONTACT_THRESHOLD_N
        lowest = tip[:, 2] + group.minimum_peg_height_offset(q)
        tips.append(tip)
        contacts.append(contact)
        surface.append(contact & (lowest > -0.001))
    return torch.cat(tips), torch.cat(contacts), torch.cat(surface)


COUNTERS = ("steps", "surface_steps", "abs_trans", "abs_rot", "trans_flips", "trans_pairs",
            "rot_flips", "rot_pairs", "contact_rot_flips", "contact_rot_pairs",
            "free_flips", "free_pairs", "free_steps", "free_force_sq",
            "velocity_reversals", "velocity_pairs", "first_trans_mm", "first_rot_deg")


@torch.inference_mode()
def evaluate_real(actor, cfg: InsertionCfg, device: str, trials_per_shape: int) -> list[dict]:
    actor.reset()
    env = InsertionEnv(cfg, device)
    if isinstance(actor, MapPolicy):
        actor.search_map = env.search_map
    n = env.num_envs
    shapes = [group.shape for group in env.groups for _ in range(group.n)]
    stats = {key: torch.zeros(n, device=device) for key in COUNTERS}
    previous_action = torch.zeros(n, 6, device=device)
    previous_tip = torch.zeros(n, 3, device=device)
    previous_velocity = torch.zeros(n, 2, device=device)
    obs = env.get_observations()
    counts = {s: 0 for s in cfg.shapes}
    records = []
    limit = env.max_episode_length * (math.ceil(trials_per_shape / cfg.envs_per_shape) + 1)
    for _ in range(limit):
        if min(counts.values()) >= trials_per_shape:
            break
        tip, contact, surface = probe(env)
        actions = actor(obs)
        started = stats["steps"] > 0
        free = started & ~contact
        velocity = tip[:, :2] - previous_tip[:, :2]
        stats["first_trans_mm"] = torch.where(
            started, stats["first_trans_mm"],
            torch.linalg.vector_norm(actions[:, :3], dim=-1) * cfg.translation_limit * 1000)
        stats["first_rot_deg"] = torch.where(
            started, stats["first_rot_deg"],
            torch.linalg.vector_norm(actions[:, 3:], dim=-1) * math.degrees(cfg.rotation_limit))
        stats["abs_trans"] += actions[:, :3].abs().mean(-1)
        stats["abs_rot"] += actions[:, 3:].abs().mean(-1)
        stats["surface_steps"] += surface.float()
        flips, pairs = sign_flips(previous_action[:, :3], actions[:, :3])
        stats["trans_flips"] += flips * started
        stats["trans_pairs"] += pairs * started
        flips, pairs = sign_flips(previous_action[:, 3:], actions[:, 3:])
        stats["rot_flips"] += flips * started
        stats["rot_pairs"] += pairs * started
        stats["contact_rot_flips"] += flips * (started & contact)
        stats["contact_rot_pairs"] += pairs * (started & contact)
        flips, pairs = sign_flips(previous_action, actions)
        stats["free_flips"] += flips * free
        stats["free_pairs"] += pairs * free
        stats["free_steps"] += free.float()
        stats["free_force_sq"] += obs["actor"][:, 7:10].square().sum(-1) * free
        reversed_, valid = velocity_reversals(previous_velocity, velocity)
        has_velocity = stats["steps"] > 1
        stats["velocity_reversals"] += (reversed_ & has_velocity).float()
        stats["velocity_pairs"] += (valid & has_velocity).float()
        stats["steps"] += 1
        previous_action, previous_tip = actions.clone(), tip
        previous_velocity = torch.where(started.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        obs, _, dones, info = env.step(actions)
        actor.reset(dones)
        terminal = info["terminal"]
        for index in dones.nonzero().flatten().tolist():
            shape = shapes[index]
            if counts[shape] < trials_per_shape:
                counts[shape] += 1
                row = {key: float(value[index]) for key, value in stats.items()}
                records.append({"shape": shape, "success": int(terminal["reason"][index]) == 1,
                                "reason": int(terminal["reason"][index]),
                                "seconds": float(terminal["length"][index]) * 0.02, **row})
            for value in stats.values():
                value[index] = 0
    else:
        raise RuntimeError("real-profile evaluation exceeded its bounded trial budget")
    return records


def summarize(records: list[dict]) -> dict:
    def ratio(a: str, b: str) -> float | None:
        total = sum(r[b] for r in records)
        return sum(r[a] for r in records) / total if total else None

    steps = sum(r["steps"] for r in records)
    good = [r["seconds"] for r in records if r["success"]]
    return {
        "trials": len(records),
        "success_rate": sum(r["success"] for r in records) / len(records),
        "mean_success_seconds": sum(good) / len(good) if good else None,
        "mean_surface_dwell_seconds": sum(r["surface_steps"] for r in records) * 0.02 / len(records),
        "mean_first_step_translation_mm": sum(r["first_trans_mm"] for r in records) / len(records),
        "mean_first_step_rotation_deg": sum(r["first_rot_deg"] for r in records) / len(records),
        "mean_abs_translation_action": sum(r["abs_trans"] for r in records) / steps,
        "mean_abs_rotation_action": sum(r["abs_rot"] for r in records) / steps,
        "translation_sign_flip_ratio": ratio("trans_flips", "trans_pairs"),
        "rotation_sign_flip_ratio": ratio("rot_flips", "rot_pairs"),
        "contact_rotation_sign_flip_ratio": ratio("contact_rot_flips", "contact_rot_pairs"),
        "free_air_sign_flip_ratio": ratio("free_flips", "free_pairs"),
        "free_air_observed_force_rms_n": (
            math.sqrt(ratio("free_force_sq", "free_steps")) if ratio("free_force_sq", "free_steps") else None),
        "xy_velocity_reversal_ratio": ratio("velocity_reversals", "velocity_pairs"),
        "terminations": {name: sum(r["reason"] == code for r in records)
                         for code, name in ((1, "success"), (2, "wrench_limit"),
                                            (3, "workspace"), (4, "timeout"),
                                            (5, "simulation_failure"))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", help="label=path/to/checkpoint.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shapes", nargs="+", default=list(TRAIN_SHAPES))
    parser.add_argument("--envs-per-shape", type=int, default=32)
    parser.add_argument("--trials", type=int, default=64, help="per shape")
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("runs/real2sim/eval_real"))
    parser.add_argument("--lowpass-off", action="store_true",
                        help="real plant without the deployment low-pass (k=650, m=0.1 kg, zeta=0.35 response)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    profile = {**REAL_PROFILE, "target_lowpass_off_probability": 1.0 if args.lowpass_off else 0.0}
    report = {"profile": {**profile, "shape_env_counts": []}, "checkpoints": {}}
    for item in args.checkpoints:
        label, path = item.split("=", 1)
        metadata, actor = load_checkpoint(Path(path), args.device)
        cfg = replace(InsertionCfg.from_checkpoint(metadata["environment"]), **profile,
                      shapes=tuple(args.shapes), envs_per_shape=args.envs_per_shape,
                      seed=args.seed, episode_seconds=args.episode_seconds,
                      physics_cuda_graph=True, observation_cuda_graph=True)
        records = evaluate_real(actor, cfg, args.device, args.trials)
        report["checkpoints"][label] = {
            "path": str(Path(path).resolve()), "overall": summarize(records),
            "shapes": {s: summarize([r for r in records if r["shape"] == s]) for s in args.shapes},
        }
        (args.output / f"trials_{label}.json").write_text(json.dumps(records) + "\n")
        print(label, json.dumps(report["checkpoints"][label]["overall"], indent=2))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
