"""冻结策略的多条件泛化评估，不按测试条件缩放动作幅度"""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from insertion import ALL_SHAPES, InsertionCfg, InsertionEnv

REASONS = {1: "success", 2: "wrench_limit", 3: "workspace", 4: "timeout"}


def load_checkpoint(path: Path, device: str) -> tuple[dict, RNNModel]:
    saved = torch.load(path, map_location=device, weights_only=True)
    metadata = saved["infos"]
    if metadata["contract"]["version"] != 1:
        raise ValueError("unsupported policy contract")
    config = dict(metadata["runner"]["actor"])
    config.pop("class_name")
    config.pop("cnn_cfg", None)
    dummy = TensorDict({"actor": torch.zeros(1, 13, device=device)}, [1])
    actor = RNNModel(dummy, {"actor": ["actor"]}, "actor", 6, **config).to(device)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    return metadata, actor


def wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    denom = 1 + z*z / trials
    center = (p + z*z / (2*trials)) / denom
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denom
    return center - half, center + half


@torch.inference_mode()
def evaluate_condition(actor: RNNModel, cfg: InsertionCfg, device: str,
                       trials_per_shape: int) -> list[dict]:
    if trials_per_shape < 1:
        raise ValueError("trials_per_shape must be positive")
    actor.reset()
    env = InsertionEnv(cfg, device)
    obs = env.get_observations()
    records = []
    counts = {s: 0 for s in cfg.shapes}
    steps = 0
    while min(counts.values()) < trials_per_shape:
        actions = actor(obs)
        obs, _, dones, info = env.step(actions)
        actor.reset(dones)
        terminal = info["terminal"]
        for index in dones.nonzero().flatten().tolist():
            shape = cfg.shapes[index // cfg.envs_per_shape]
            if counts[shape] >= trials_per_shape:
                continue
            counts[shape] += 1
            reason = REASONS[int(terminal["reason"][index])]
            records.append({
                "shape": shape, "seed": cfg.seed, "clearance_m": cfg.clearance,
                "scale": cfg.scale, "dynamics": cfg.dynamics,
                "episode_budget_seconds": cfg.episode_seconds,
                "fixed_target_depth_m": cfg.fixed_target_depth_m,
                "target_depth_m": float(terminal["target_depth_m"][index]),
                "trial": counts[shape], "success": int(reason == "success"),
                "seconds": float(terminal["length"][index]) * 0.02,
                "peak_force_n": float(terminal["peak_force"][index]),
                "peak_moment_nm": float(terminal["peak_moment"][index]),
                "termination": reason,
            })
        steps += 1
        if steps > env.max_episode_length * (math.ceil(trials_per_shape/cfg.envs_per_shape)+1):
            raise RuntimeError("evaluation exceeded its bounded trial budget")
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in records:
        key = (row["shape"], row["clearance_m"], row["scale"], row["dynamics"],
               row["episode_budget_seconds"], row["fixed_target_depth_m"])
        groups.setdefault(key, []).append(row)
    result = []
    for (shape, clearance, scale, dynamics, budget, fixed_depth), rows in groups.items():
        n, successes = len(rows), sum(r["success"] for r in rows)
        low, high = wilson(successes, n)
        good = [r["seconds"] for r in rows if r["success"]]
        result.append({
            "shape": shape, "clearance_m": clearance, "scale": scale, "dynamics": dynamics,
            "episode_budget_seconds": budget,
            "fixed_target_depth_m": fixed_depth,
            "sampled_target_depth_min_m": min(r["target_depth_m"] for r in rows),
            "sampled_target_depth_max_m": max(r["target_depth_m"] for r in rows),
            "trials": n, "successes": successes, "success_rate": successes/n,
            "wilson_95": [low, high],
            "mean_success_seconds": sum(good)/len(good) if good else None,
            "max_force_n": max(r["peak_force_n"] for r in rows),
            "max_moment_nm": max(r["peak_moment_nm"] for r in rows),
            "terminations": {v: sum(r["termination"] == v for r in rows) for v in REASONS.values()},
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shapes", nargs="+", choices=ALL_SHAPES, default=list(ALL_SHAPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 2026])
    parser.add_argument("--clearances", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--dynamics", nargs="+", choices=["train", "nominal", "soft", "stiff"],
                        default=["nominal", "soft", "stiff"])
    parser.add_argument("--envs-per-shape", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--target-depths", nargs="+", type=float,
                        help="fixed evaluation depths in metres; default follows checkpoint")
    parser.add_argument("--trials", type=int, default=16, help="per shape, condition and seed")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args()
    if any(c < 0.0001 for c in args.clearances):
        parser.error("sub-0.1 mm tests require a separate validated solver-accuracy study")
    torch.set_num_threads(1)
    metadata, actor = load_checkpoint(args.checkpoint, args.device)
    base = InsertionCfg.from_checkpoint(metadata["environment"])
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for clearance in args.clearances:
        for scale in args.scales:
            for dynamics in args.dynamics:
                for depth in (args.target_depths if args.target_depths is not None
                              else [base.fixed_target_depth_m]):
                    for seed in args.seeds:
                        cfg = replace(base, shapes=tuple(args.shapes), clearance=clearance,
                                      scale=scale, dynamics=dynamics, seed=seed, training=False,
                                      fixed_target_depth_m=depth,
                                      envs_per_shape=args.envs_per_shape,
                                      episode_seconds=(base.episode_seconds if args.episode_seconds is None
                                                       else args.episode_seconds))
                        rows = evaluate_condition(actor, cfg, args.device, args.trials)
                        records.extend(rows)
                        print(json.dumps(summarize(rows), indent=2))
    with (args.output / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = summarize(records)
    for item in summaries:
        item["unseen_geometry"] = item["shape"] not in base.shapes
        item["unseen_scale"] = item["scale"] != base.scale
        item["unseen_clearance"] = item["clearance_m"] != base.clearance
    report = {"checkpoint": str(args.checkpoint.resolve()),
              "training_environment": metadata["environment"], "conditions": summaries,
              "worst_group_success_rate": min(r["success_rate"] for r in summaries)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
