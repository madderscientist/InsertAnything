"""真机 real2sim 适配续训：从已部署 checkpoint 迁移配置，套用 REAL2SIM_VALUES 后交给 train.py 持续训练

用法（仓库根目录）:
    python -m real2sim.train_real2sim --output runs/real2sim/stage3_from_961
"""

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import logging
from pathlib import Path
import shutil

import torch

from insertion import InsertionCfg
from insertion.config import REAL2SIM_FIELDS, REAL2SIM_VALUES
from insertion.ppo import SMOOTH_PPO
from train import atomic_json, contract, supervise, validate_ppo_batch_layout

DEFAULT_SOURCE = Path("runs/releases/stage2-outer-xy-best-961/checkpoint.pt")
# 训练长期被自适应 KL 压在 1e-5 学习率下限；悬空力旋量扰动取实测灵敏度量级
ALGORITHM_OVERRIDES = dict(class_name=SMOOTH_PPO, desired_kl=0.02, entropy_coef=0.0,
                           smooth_coef=0.05, smooth_force_std=0.5, smooth_moment_std=0.05)
ACTION_STD_MAX = 0.4


def real2sim_config(source_environment: dict) -> InsertionCfg:
    defaults = InsertionCfg()
    source = InsertionCfg.from_checkpoint(source_environment)
    # 旧检查点可能带 800 的安全罚分，统一改用当前默认值
    return replace(source, **REAL2SIM_VALUES, safety_failure_penalty=defaults.safety_failure_penalty)


def prepare(source: Path, output: Path, device: str, envs_per_shape: int | None = None,
            steps: int | None = None) -> None:
    saved = torch.load(source, map_location="cpu", weights_only=True)
    infos = saved["infos"]
    cfg = real2sim_config(infos["environment"])
    if envs_per_shape is not None:
        cfg = replace(cfg, envs_per_shape=envs_per_shape, shape_env_counts=())
    runner = copy.deepcopy(infos["runner"])
    runner["save_interval"] = 1
    runner["algorithm"].update(ALGORITHM_OVERRIDES)
    std_range = runner["actor"]["distribution_cfg"]["std_range"]
    runner["actor"]["distribution_cfg"]["std_range"] = [std_range[0], ACTION_STD_MAX]
    if steps is not None:
        runner["num_steps_per_env"] = steps
    metadata = {
        "environment": asdict(cfg), "runner": runner, "contract": contract(cfg),
        "simulation_devices": [device], "collection_backend": "serial",
        "real2sim_source": {
            "checkpoint": str(source.resolve()), "iteration": saved["iter"],
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "applied_fields": list(REAL2SIM_FIELDS),
        },
    }
    validate_ppo_batch_layout(metadata, distributed=False)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", metadata)
    # 保留源 infos.environment，使续训检测到配置变化并清空成功率窗口
    shutil.copyfile(source, output / "checkpoint.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path("runs/real2sim/stage3_from_961"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs-per-shape", type=int, help="override the source layout (prepare only)")
    parser.add_argument("--steps", type=int, help="override rollout steps per environment (prepare only)")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.output / "config.json").exists():
        prepare(args.source, args.output, args.device, args.envs_per_shape, args.steps)
    if not args.prepare_only:
        supervise(argparse.Namespace(output=args.output, resume=True, device=args.device,
                                     check_seconds=300., stall_seconds=900., stop_grace_seconds=180.))


if __name__ == "__main__":
    main()
