"""随机形状插孔策略从零训练：双卡分布式 PPO，初始化与任务同 main，只换成随机尺寸的插头与插孔

用法（仓库根目录）:
    python -m search_prior.train_search --output runs/search/baseline
"""

import argparse
from dataclasses import asdict, replace
import logging
import os
from pathlib import Path

# 更新批次大小随轨迹长度变化，默认分配器碎片化导致频繁 OOM 重试；须在 CUDA 初始化前设置，worker 子进程继承
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from insertion import InsertionCfg
from insertion.config import REAL2SIM_VALUES, runner_config
from insertion.model import GSDE_DISTRIBUTION, SKIP_RNN_MODEL
from train import atomic_json, contract, supervise, validate_ppo_batch_layout

SEARCH_SHAPES = ("triangle", "pentagon", "hexagon", "l", "rectangle", "dual")
SEARCH_ACTOR_MIN_LR = 1e-6


def parse_critic_lr(value: str) -> float | str:
    return value if value == "actor" else float(value)


def search_config(args: argparse.Namespace) -> InsertionCfg:
    return replace(
        replace(InsertionCfg(), **REAL2SIM_VALUES),
        # 停滞惩罚只在板面受阻时触发，空中漂移可规避，从零训练会退化为远离板面游荡
        stagnation_cost_per_second=0.0,
        # 目标上叠加的正弦粘滑空中也一直存在，与真机不符
        stick_slip_max_pp_m=0.0,
        wall_friction_force=2.0,
        # 线性变化成本压不住悬空大幅跳变，低阻尼下激起持续振荡
        free_air_action_change_squared=True,
        # ζ=0.25 超调 44%，满难度下悬空惯性力 p90 近 5 N；0.5 超调 16%
        damping_ratio_min=0.5,
        clearance_curriculum=True, noise_curriculum=True, noise_curriculum_mode="adr",
        physics_cuda_graph=True, observation_cuda_graph=True,
        shapes=SEARCH_SHAPES, shape_randomization=True, rim_probability=args.rim_probability,
        episode_seconds=args.episode_seconds,
        envs_per_shape=args.envs_per_shape, shape_env_counts=(), adaptive_sampling=True)


def search_runner(args: argparse.Namespace) -> dict:
    runner = runner_config(args.steps)
    runner["save_interval"] = 1
    layers = 2 if args.rnn_layers is None else args.rnn_layers
    # 部分可观测下 critic 也需记忆：只看当前特权状态的 MLP critic 训练均崩塌
    for name in ("actor", "critic"):
        runner[name].update(class_name=SKIP_RNN_MODEL, rnn_hidden_dim=args.rnn_hidden, rnn_num_layers=layers,
                            hidden_dims=(args.rnn_hidden, args.rnn_hidden // 2))
    if getattr(args, "critic_rnn_layers", None) is not None:
        runner["critic"]["rnn_num_layers"] = args.critic_rnn_layers
    distribution = runner["actor"]["distribution_cfg"]
    runner["actor"]["distribution_cfg"] = {
        "class_name": GSDE_DISTRIBUTION, "init_std": distribution["init_std"],
        "std_range": distribution["std_range"], "resample_steps": args.gsde_resample_steps}
    # 每卡环境数 × 序列长度 × 256 维 GRU 的更新显存较大，需更小的 minibatch
    runner["algorithm"].update(num_mini_batches=args.mini_batches,
                               phase_coef=args.phase_coef, phase_learning_rate=args.phase_lr)
    runner["algorithm"].update(actor_min_lr=getattr(args, "actor_min_lr", SEARCH_ACTOR_MIN_LR),
                               critic_learning_rate=getattr(args, "critic_lr", "actor"))
    return runner


def prepare(args: argparse.Namespace) -> None:
    devices = [f"cuda:{i}" for i in range(args.gpus)]
    cfg = search_config(args)
    metadata = {
        "environment": asdict(cfg), "runner": search_runner(args), "contract": contract(cfg),
        "simulation_devices": devices,
        "collection_backend": "distributed" if len(devices) > 1 else "serial",
    }
    validate_ppo_batch_layout(metadata, distributed=len(devices) > 1)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "config.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("runs/search/baseline"))
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--envs-per-shape", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128,
                        help="rollout steps per environment; also the truncated BPTT horizon")
    parser.add_argument("--mini-batches", type=int, default=16)
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, help="override both GRU depths (default: 2)")
    parser.add_argument("--critic-rnn-layers", type=int, help="override the critic GRU depth independently")
    parser.add_argument("--actor-min-lr", type=float, default=SEARCH_ACTOR_MIN_LR,
                        help="lower bound for the actor KL learning-rate schedule")
    parser.add_argument("--critic-lr", type=parse_critic_lr, default="actor", metavar="RATE|actor",
                        help="follow the actor KL schedule (default), or specify a fixed critic learning rate")
    parser.add_argument("--gsde-resample-steps", type=int, default=25,
                        help="gSDE exploration matrix resample period in policy steps")
    parser.add_argument("--phase-coef", type=float, default=0.1,
                        help="phase cross-entropy weight on the actor GRU (default: 0.1, use 0 to disable)")
    parser.add_argument("--phase-lr", type=float, default=1e-3,
                        help="fixed Adam learning rate of the phase head, independent of the KL schedule")
    parser.add_argument("--rim-probability", type=float, default=0.5,
                        help="per-episode probability of a raised rim around the opening")
    parser.add_argument("--episode-seconds", type=float, default=InsertionCfg.episode_seconds)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.output / "config.json").exists():
        prepare(args)
    if not args.prepare_only:
        supervise(argparse.Namespace(output=args.output, resume=True, device="cuda:0",
                                     check_seconds=300., stall_seconds=900., stop_grace_seconds=180.))


if __name__ == "__main__":
    main()
