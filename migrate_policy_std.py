"""Migrate a paused Gaussian policy to state-dependent std in a new run."""

import argparse
import copy
from dataclasses import asdict, replace
import math
from pathlib import Path

import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from insertion import InsertionCfg
from train import atomic_json, contract, prepare_runner_config


def migrate_checkpoint(saved: dict, initial_std: float | None = None,
                       noise_level: int | None = None) -> dict:
    result = copy.deepcopy(saved)
    metadata = result["infos"]
    config = prepare_runner_config(metadata["runner"])["actor"]
    config.pop("class_name")
    distribution = config["distribution_cfg"]
    if distribution["class_name"] != "GaussianDistribution" or distribution["std_type"] != "log":
        raise ValueError("migration requires a state-independent log Gaussian policy")
    if initial_std is not None and (
            not math.isfinite(initial_std)
            or not distribution["std_range"][0] <= initial_std <= distribution["std_range"][1]):
        raise ValueError("initial_std must be finite and within the source std_range")
    cfg = replace(InsertionCfg.from_checkpoint(metadata["environment"]), shape_env_counts=())
    if noise_level is not None and (
            type(noise_level) is not int or not 0 <= noise_level <= 10
            or not cfg.noise_curriculum or cfg.noise_curriculum_mode != "success"):
        raise ValueError("noise_level requires a success noise curriculum and an integer in [0, 10]")
    obs = TensorDict({"actor": torch.zeros(1, 13)}, [1])
    old = RNNModel(obs, {"actor": ["actor"]}, "actor", 6, **copy.deepcopy(config))
    old.load_state_dict(saved["actor_state_dict"], strict=True)
    distribution["class_name"] = "HeteroscedasticGaussianDistribution"
    distribution.pop("learn_std", None)
    new = RNNModel(obs, {"actor": ["actor"]}, "actor", 6, **copy.deepcopy(config))
    old_parameters = dict(old.named_parameters())
    new_parameters = dict(new.named_parameters())
    head, output_layer = [(name, layer) for name, layer in old.named_modules()
                          if name.startswith("mlp.") and isinstance(layer, torch.nn.Linear)][-1]
    if output_layer.out_features != 6:
        raise ValueError("migration requires a six-action output layer")
    head_parameters = (f"{head}.weight", f"{head}.bias")
    state = new.state_dict()
    log_std = old.distribution.log_std_param.detach().clamp(*old.distribution.log_std_range)
    if initial_std is not None:
        log_std = torch.full_like(log_std, math.log(initial_std))
        distribution["init_std"] = initial_std
    for name, value in old.state_dict().items():
        if name == "distribution.log_std_param":
            continue
        if name in head_parameters:
            state[name][:6].copy_(value)
            state[name][6:].zero_()
        else:
            state[name].copy_(value)
    state[f"{head}.bias"][6:].copy_(log_std)
    new.load_state_dict(state, strict=True)
    result["actor_state_dict"] = state

    # PPO orders actor parameters before critic parameters in its single optimizer group.
    optimizer = result["optimizer_state_dict"]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("migration requires a single PPO optimizer parameter group")
    group = optimizer["param_groups"][0]
    ids = group["params"]
    if len(ids) < len(old_parameters):
        raise ValueError("optimizer does not contain all actor parameters")
    actor_ids = dict(zip(old_parameters, ids[:len(old_parameters)], strict=True))
    optimizer["state"].pop(actor_ids["distribution.log_std_param"], None)
    for name in head_parameters:
        for key, value in optimizer["state"].get(actor_ids[name], {}).items():
            if isinstance(value, torch.Tensor) and value.shape == old_parameters[name].shape:
                expanded = torch.zeros_like(new_parameters[name])
                expanded[:6].copy_(value)
                optimizer["state"][actor_ids[name]][key] = expanded
    group["params"] = [actor_ids[name] for name in new_parameters] + ids[len(old_parameters):]

    metadata["runner"]["actor"]["distribution_cfg"] = copy.deepcopy(distribution)
    metadata.update(environment=asdict(cfg), contract=contract(cfg))
    level = 0 if noise_level is None else len(cfg.active_clearance_levels)-1+noise_level
    metadata["noise_curriculum_state"] = {
        "level": level, "windows": {shape: [] for shape in cfg.shapes}}
    metadata.pop("adaptive_sampling_state", None)
    metadata["std_migration"] = {
        "source_iteration": saved["iter"], "curriculum_reset": True,
        "curriculum_start_level": level,
        "initial_std": log_std.exp().tolist(),
        "std_range": list(distribution["std_range"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-std", type=float,
                        help="initialize every std output to this value instead of inheriting it")
    parser.add_argument("--noise-level", type=int,
                        help="start at target clearance and this noise level (0-10); default: full curriculum reset")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    migrated = migrate_checkpoint(saved, args.initial_std, args.noise_level)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(migrated, args.output / "checkpoint.pt")
    atomic_json(args.output / "config.json", migrated["infos"])
    level = migrated["infos"]["noise_curriculum_state"]["level"]
    print(f"Migrated iteration {saved['iter']} to {args.output}; curriculum start level {level}.")


if __name__ == "__main__":
    main()
