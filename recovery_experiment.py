"""Opt-in, bounded matched-control recovery experiment; never modifies baseline runs."""

import argparse
from collections import deque
import copy
from dataclasses import asdict, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch

from evaluate import REASONS, load_checkpoint, wilson
from insertion import ALL_SHAPES, InsertionCfg, InsertionEnv
from insertion.transforms import rotate, tip_pose
from insertion.env import SimulationCfg
from train import atomic_json, contract, make_runner, restore_environment_progress


VERSION = 1
OFFSETS = ((0.0005, 0.001), (0.001, 0.002), (0.002, 0.003))
SOURCE = Path("runs/continuous_no_intent/corner_entry_checkpoint.pt")
MOTION = Path("runs/continuous_no_intent/recovery_motion_19610/summary.json")
SIMULATION_CAPACITY = {"nconmax": 512, "njmax": 2048}
TAIL_SAMPLES = 101


class BoundedStop(Exception):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def clear_stale_error(state):
    if state.get("error") is not None:
        state.setdefault("error_history", []).append({
            "error": state.pop("error"), "status": state.get("status"),
            "current_job": state.get("current_job"), "cleared": time.time()})


def atomic_checkpoint(path, saved):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    torch.save(saved, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def motion_gate(path, source_hash, source=SOURCE):
    """Consume the diagnostic's actual gate without weakening its population thresholds."""
    if not Path(path).is_file():
        return {"allowed": False, "reason": "motion diagnostic not ready"}
    report = read_json(path)
    gate = report.get("curriculum_gate", {})
    eligible = gate.get("eligible_blocked_failures", 0)
    fraction = gate.get("fraction_actual_radius_lt_0p5mm", 0.)
    checkpoint_matches = (report.get("checkpoint") == str(Path(source).resolve())
                          and digest(source) == source_hash)
    allowed = (gate.get("pilot_justified") is True and type(eligible) is int and eligible >= 30
               and isinstance(fraction, (int, float)) and math.isfinite(fraction)
               and .5 <= fraction <= 1. and report.get("noise_scale") == 1.
               and checkpoint_matches)
    return {"allowed": bool(allowed),
            "reason": ("diagnostic supports controlled pilot" if allowed else
                       "diagnostic gate not satisfied: require explicit pilot_justified, "
                       ">=30 eligible blocked failures, >=50% actual 1s radii <0.5mm, "
                       "full noise and matching frozen checkpoint"),
            "evidence": gate, "gate_definition": report.get("gate_definition"),
            "checkpoint_matches": checkpoint_matches,
            "diagnostic": str(Path(path).resolve()), "sha256": digest(path)}


def hypothesis_gate(path, source_hash, source, hypothesis="small-motion"):
    original = motion_gate(path, source_hash, source)
    if hypothesis == "small-motion":
        return {**original, "hypothesis": hypothesis}
    if hypothesis != "oscillatory-recovery":
        raise ValueError("unknown experimental hypothesis")
    sensitivity = Path(path).with_name("sensitivity.json")
    if not sensitivity.is_file():
        return {"allowed": False, "hypothesis": hypothesis,
                "reason": "oscillation sensitivity evidence missing", "original_gate": original}
    if not Path(path).is_file():
        return {"allowed": False, "hypothesis": hypothesis,
                "reason": "source/full-noise diagnostic missing", "original_gate": original}
    evidence = read_json(sensitivity)
    tail = evidence.get("mean_actions", {}).get("timeout_tail_2s", {})
    values = {
        "windows": tail.get("windows"),
        "radius_mm": tail.get("actual", {}).get("radius_mm", {}).get("median"),
        "reversals": tail.get("actual", {}).get("direction_reversals_over_120deg", {}).get("median"),
        "depth_gain_mm": tail.get("depth_gain_mm", {}).get("median")}
    valid = all(type(v) in (int, float) and math.isfinite(v) for v in values.values())
    diagnostic = read_json(path)
    allowed = (valid and values["windows"] >= 60 and values["radius_mm"] >= .5
               and values["reversals"] >= 10 and values["depth_gain_mm"] <= .1
               and original.get("checkpoint_matches", False) and diagnostic.get("noise_scale") == 1.)
    return {"allowed": bool(allowed), "hypothesis": hypothesis, "original_gate": original,
            "reason": ("prospective exploratory oscillatory-recovery pilot eligible" if allowed
                       else "oscillatory-recovery eligibility not satisfied"),
            "thresholds": {"windows_min": 60, "median_radius_mm_min": .5,
                           "median_reversals_min": 10, "median_depth_gain_mm_max": .1},
            "evidence": values, "sensitivity": str(sensitivity.resolve()),
            "sha256": digest(sensitivity),
            "interpretation": "Ineffective oscillatory search supports testing near-mouth starts; "
                              "not causal proof, not small-motion gate success or std enlargement."}


def experiment_configs(saved, seed, *, training=True, envs_per_shape=None):
    base = InsertionCfg.from_checkpoint(saved["infos"]["environment"])
    changes = dict(seed=seed, training=training, noise_curriculum=False,
                   noise_curriculum_mode="steps", clearance_curriculum=False,
                   adaptive_sampling=False)
    if envs_per_shape is not None:
        changes.update(envs_per_shape=envs_per_shape, shape_env_counts=())
    if not training:
        changes.update(shapes=ALL_SHAPES, shape_env_counts=())
    cfg = replace(base, **changes)
    runner = copy.deepcopy(saved["infos"]["runner"])
    runner["seed"] = seed
    runner["algorithm"]["entropy_coef"] = 0.001
    runner["save_interval"] = 1
    runner["check_for_nan"] = True
    return cfg, runner


def recovery_pose(group, q, generator, stage):
    n = len(q)
    u = torch.rand(n, 3, generator=generator, device=q.device)
    low, high = OFFSETS[stage]
    radius = low + (high - low) * u[:, 0]
    angle = 2 * math.pi * u[:, 1]
    margin = 0.0002 + 0.0003 * u[:, 2]
    return torch.stack((radius * angle.cos(), radius * angle.sin(),
                        margin - group.minimum_peg_height_offset(q)), dim=-1)


def install_recovery_reset(group, fraction, stage, seed):
    if not 0 <= fraction <= 1 or not 0 <= stage < len(OFFSETS):
        raise ValueError("invalid recovery reset recipe")
    original = group.reset
    generator = torch.Generator(device=group.device).manual_seed(seed)
    group.recovery_designated = torch.zeros(group.n, dtype=torch.bool, device=group.device)
    group.recovery_reset_counts = [0, 0]

    def reset(self, ids):
        original(ids)
        self.recovery_designated[ids] = False
        if fraction == 0 or len(ids) == 0:
            self.recovery_reset_counts[0] += len(ids)
            return
        mask = torch.rand(len(ids), generator=generator, device=self.device) < fraction
        selected = ids[mask]
        self.recovery_designated[selected] = True
        self.recovery_reset_counts[0] += len(ids) - len(selected)
        self.recovery_reset_counts[1] += len(selected)
        if len(selected) == 0:
            return
        q = self.sim.data.qpos[selected, 3:7]
        tip = recovery_pose(self, q, generator, stage)
        self.sim.data.qpos[selected, :3] = tip - rotate(q, self.offset[selected])
        self.sim.forward()
        nominal = tip_pose(self.sim.data.qpos[selected, :3], q,
                           self.nominal_offset[selected] + self.calibration[selected])
        self.targets_p[:, selected] = nominal.unsqueeze(0)
        self.targets_q[:, selected] = q.unsqueeze(0)

    group.reset = MethodType(reset, group)


class RecoveryEnv(InsertionEnv):
    def __init__(self, cfg, device, fraction=0., stage=0, saved=None, attempt=0):
        def capacity_config(**kwargs):
            kwargs.update(SIMULATION_CAPACITY)
            return SimulationCfg(**kwargs)

        # Near-mouth contact can exceed baseline allocation; capacity is not a physics change.
        with patch("insertion.env.SimulationCfg", side_effect=capacity_config):
            super().__init__(cfg, device)
        # Restore progress through the supported API, then reset at full physical difficulty.
        if saved is not None:
            restore_environment_progress(self, cfg, saved, attempt)
        for index, group in enumerate(self.groups):
            group.noise_scale = group.difficulty = 1.
            install_recovery_reset(group, fraction, stage, cfg.seed + index * 10007 + attempt + 910019)
            group.reset(torch.arange(group.n, device=group.device))
        self.policy_steps = max(self.policy_steps, cfg.curriculum_steps)
        self.last_motion = None
        for group in self.groups:
            step = group.step

            def observed_step(actions, group=group, original=step):
                result = original(actions)
                position, quaternion, _, _ = group.state()
                group.experiment_motion = (
                    tip_pose(position, quaternion, group.offset).clone(),
                    torch.linalg.vector_norm(group.sim.data.qfrc_constraint[:, :3], dim=-1).clone())
                return result

            group.step = observed_step

    def step(self, actions):
        for group in self.groups:
            group.difficulty = group.noise_scale = 1.
        result = super().step(actions)
        # Group steps capture physical terminal states before InsertionEnv auto-resets.
        self.last_motion = (torch.cat([g.experiment_motion[0] for g in self.groups]),
                            torch.cat([g.experiment_motion[1] for g in self.groups]))
        return result


def trajectory_metrics(positions, forces):
    """Secondary outcomes over at most the final two seconds, terminal step included."""
    p = np.asarray(positions, dtype=float)[-TAIL_SAMPLES:]
    f = np.asarray(forces, dtype=float)[-len(p):]
    delta = np.diff(p[:, :2], axis=0)
    norm = np.linalg.norm(delta, axis=1)
    meaningful = delta[norm >= .000005]
    lengths = np.linalg.norm(meaningful, axis=1)
    cos = (meaningful[:-1] * meaningful[1:]).sum(axis=1) / np.maximum(lengths[:-1] * lengths[1:], 1e-30)
    gain = float((p[0, 2] - p[-1, 2]) * 1000)
    contact = float(np.mean(f > 1.))
    return {"tail_steps": len(p), "tail_depth_gain_mm": gain,
            "tail_radius_mm": float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max() * 1000),
            "tail_path_mm": float(norm.sum() * 1000),
            "tail_reversals_over_120deg": int(np.sum(cos < -.5)),
            "tail_contact_fraction_gt1n": contact,
            "tail_nonprogress": int(gain <= .1),
            "tail_contact_nonprogress": int(contact >= .2 and gain <= .1)}


def finite_tree(value, name="state"):
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite {name}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_tree(item, f"{name}.{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"nonfinite {name}")


def release_update_cache(device):
    if not str(device).startswith("cuda"):
        return {}
    with torch.cuda.device(device):
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        return {"allocated_bytes": allocated, "reserved_before_bytes": reserved,
                "reserved_after_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": peak, "free_bytes": free, "total_bytes": total}


def train_worker(job):
    seed, arm, target = job["seed"], job["arm"], job["target"]
    checkpoint = Path(job["checkpoint"])
    resume = checkpoint.exists()
    saved = torch.load(checkpoint if resume else job["source"],
                       map_location=job["device"], weights_only=True)
    completed = saved["infos"].get("recovery_experiment", {}).get("updates", 0) if resume else 0
    if completed >= target:
        return
    cfg, config = experiment_configs(saved, seed, envs_per_shape=job["train_envs"])
    torch.manual_seed(seed + completed)
    env = RecoveryEnv(cfg, job["device"], 0.25 if arm == "recovery" else 0.,
                      job["stage"], saved, completed)
    runner = make_runner(env, config, str(checkpoint.parent / "logs"))
    runner.load(str(checkpoint if resume else job["source"]), map_location=job["device"])
    initial_iteration = saved["iter"] + 1
    runner.current_learning_iteration = initial_iteration
    original_update = runner.alg.update

    def checked_update():
        losses = original_update()
        finite_tree(losses, "PPO losses")
        return losses

    runner.alg.update = checked_update
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    metadata = {**saved["infos"], "environment": asdict(cfg), "runner": config,
                "contract": contract(cfg)}
    metadata.pop("noise_curriculum_state", None)
    metadata.pop("adaptive_sampling_state", None)
    last_saved = [-1]

    def save(self, path, infos=None):
        del path, infos
        updates = completed + self.current_learning_iteration - initial_iteration + 1
        if updates != last_saved[0]:
            state = self.alg.save()
            finite_tree(state)
            metadata["progress"] = {"policy_steps": env.policy_steps,
                                    "transitions": env.total_transitions}
            metadata["recovery_experiment"] = {
                "version": VERSION, "source_sha256": job["source_sha256"], "seed": seed,
                "arm": arm, "stage": job["stage"], "updates": updates,
                "hypothesis": job.get("hypothesis", "small-motion"),
                "gate_identity": job.get("gate_identity"),
                "simulation_capacity": SIMULATION_CAPACITY,
                "validation_seeds": job.get("eval_seeds"),
                "final_test_seeds": job.get("final_seeds"),
                "episode_resume": "fresh episodes and zero actor/critic GRU at worker boundary",
                "reset_counts": {g.shape: g.recovery_reset_counts for g in env.groups}}
            state.update(iter=self.current_learning_iteration, infos=metadata)
            atomic_checkpoint(checkpoint, state)
            memory = release_update_cache(job["device"])
            atomic_json(checkpoint.parent / "progress.json",
                        {**metadata["recovery_experiment"], "timestamp": time.time(),
                         "iteration": self.current_learning_iteration, "pid": os.getpid(),
                         "cuda_memory": memory})
            last_saved[0] = updates
        if stop[0] or time.time() >= job["deadline"]:
            raise BoundedStop("training stopped at a completed PPO update")

    runner.save = MethodType(save, runner)
    try:
        runner.learn(target - completed)
    finally:
        runner.logger.stop_logging_writer()


@torch.inference_mode()
def evaluate_worker(job):
    metadata, actor = load_checkpoint(Path(job["checkpoint"]), job["device"])
    saved = {"infos": metadata}
    records = []
    for condition in ("ordinary", "recovery"):
        for stage in (range(len(OFFSETS)) if condition == "recovery" else [0]):
            for seed in job["eval_seeds"]:
                cfg, _ = experiment_configs(saved, seed, training=False,
                                            envs_per_shape=job["eval_envs"])
                torch.manual_seed(seed)
                env = RecoveryEnv(cfg, job["device"], float(condition == "recovery"), stage)
                actor.reset()
                obs = env.get_observations()
                shapes = [g.shape for g in env.groups for _ in range(g.n)]
                counts = dict.fromkeys(cfg.shapes, 0)
                positions = [deque(maxlen=TAIL_SAMPLES) for _ in shapes]
                forces = [deque(maxlen=TAIL_SAMPLES) for _ in shapes]
                cap = env.max_episode_length * (math.ceil(job["trials"] / min(cfg.environment_counts)) + 2)
                for _ in range(cap):
                    if time.time() >= job["deadline"]:
                        raise BoundedStop("evaluation walltime bound")
                    obs, rewards, dones, info = env.step(actor(obs))
                    finite_tree(rewards, "evaluation rewards")
                    finite_tree(obs["actor"], "evaluation observations")
                    actor.reset(dones)
                    motion, force = (value.cpu().numpy() for value in env.last_motion)
                    for index in range(len(shapes)):
                        positions[index].append(motion[index].copy())
                        forces[index].append(float(force[index]))
                    terminal = info["terminal"]
                    for index in dones.nonzero().flatten().tolist():
                        shape = shapes[index]
                        if counts[shape] >= job["trials"]:
                            positions[index].clear()
                            forces[index].clear()
                            continue
                        counts[shape] += 1
                        reason = REASONS[int(terminal["reason"][index])]
                        row = {"condition": condition, "stage": stage, "shape": shape, "seed": seed,
                               "success": int(reason == "success"),
                               "safety": int(reason in ("wrench_limit", "workspace")),
                               "termination": reason,
                               "peak_force_n": float(terminal["peak_force"][index]),
                               "peak_moment_nm": float(terminal["peak_moment"][index]),
                               "seconds": float(terminal["length"][index]) * 0.02}
                        row.update(trajectory_metrics(positions[index], forces[index]))
                        positions[index].clear()
                        forces[index].clear()
                        finite_tree(row)
                        records.append(row)
                    if min(counts.values()) >= job["trials"]:
                        break
                else:
                    raise RuntimeError("bounded evaluation failed to complete all trials")
                del env
                if job["device"].startswith("cuda"):
                    torch.cuda.empty_cache()
    atomic_json(Path(job["output"]), {
        "checkpoint_sha256": digest(job["checkpoint"]), "trials_per_shape_per_seed": job["trials"],
        "eval_seeds": job["eval_seeds"], "evaluation_role": job.get("evaluation_role", "validation"),
        "hypothesis": job.get("hypothesis"), "gate_identity": job.get("gate_identity"),
        "simulation_capacity": SIMULATION_CAPACITY,
        "records": records, "metrics": metrics(records)})


def rate(rows, key):
    n, k = len(rows), sum(row[key] for row in rows)
    if not n:
        raise ValueError("cannot estimate an empty cohort")
    return {"n": n, "count": k, "rate": k / n, "wilson_95": list(wilson(k, n))}


def metrics(records):
    result = {}
    quantiles = torch.tensor([.5, .95, .99, 1.], dtype=torch.float64)
    keys = sorted({(r["condition"], r["stage"]) for r in records})
    for condition, stage in keys:
        cohort = [r for r in records if (r["condition"], r["stage"]) == (condition, stage)]
        for shape in ["all", *sorted({r["shape"] for r in cohort})]:
            rows = cohort if shape == "all" else [r for r in cohort if r["shape"] == shape]
            force = torch.tensor([r["peak_force_n"] for r in rows], dtype=torch.float64)
            moment = torch.tensor([r["peak_moment_nm"] for r in rows], dtype=torch.float64)
            item = {
                "success": rate(rows, "success"), "safety": rate(rows, "safety"),
                "force_quantiles_n": torch.quantile(force, quantiles).tolist(),
                "moment_quantiles_nm": torch.quantile(moment, quantiles).tolist(),
                "contact_fraction_gt_1n": float((force > 1.).double().mean()),
                "terminations": {reason: sum(r["termination"] == reason for r in rows)
                                 for reason in REASONS.values()}}
            secondary = [r for r in rows if r.get("tail_steps", 0) >= 100]
            item["secondary_full_2s_tail"] = {
                "eligible_episodes": len(secondary),
                "scope": "terminal-inclusive final <=2s; true constraint force >1N contact; "
                         "reversals skip <0.005mm segments; secondary outcomes, not reward or promotion",
                "medians": {key: float(np.median([r[key] for r in secondary])) for key in (
                    "tail_depth_gain_mm", "tail_radius_mm", "tail_path_mm",
                    "tail_reversals_over_120deg", "tail_contact_fraction_gt1n")} if secondary else {},
                "contact_nonprogress_rate": (sum(r["tail_contact_nonprogress"] for r in secondary)
                                             / len(secondary)) if secondary else None}
            result[f"{condition}/{stage}/{shape}"] = item
    return result


def difference(candidate, reference):
    # Conservative unpaired interval; do not pretend same-reset seeds imply paired outcomes.
    return {"point": candidate["rate"] - reference["rate"],
            "interval": [candidate["wilson_95"][0] - reference["wilson_95"][1],
                         candidate["wilson_95"][1] - reference["wilson_95"][0]]}


def assess(candidate, control, baseline, stage):
    comparisons = {}
    regression, certified = False, True
    for key, item in candidate.items():
        condition, level, shape = key.split("/")
        if condition == "recovery" and int(level) != stage:
            continue
        for reference_name, reference in (("control", control), ("frozen", baseline)):
            success = difference(item["success"], reference[key]["success"])
            safety = difference(item["safety"], reference[key]["safety"])
            comparisons[f"{key}/{reference_name}"] = {"success": success, "safety": safety}
            regression |= (safety["interval"][0] > .01
                           or (condition == "ordinary" and success["interval"][1] < -.02))
            certified &= safety["interval"][1] <= .01
            if condition == "ordinary":
                certified &= success["interval"][0] >= -.02
            else:
                certified &= success["interval"][0] > 0 and item["success"]["wilson_95"][0] >= .8
    key = f"recovery/{stage}/all"
    promising = (candidate[key]["success"]["rate"] > control[key]["success"]["rate"]
                 and candidate[key]["success"]["rate"] > baseline[key]["success"]["rate"])
    return {"regression": bool(regression), "certified": bool(certified),
            "promising": bool(promising), "comparisons": comparisons,
            "uncertainty": "pointwise conservative differences of Wilson 95% intervals; "
                           "exploratory repeated looks, not familywise confirmatory inference"}


def run_job(job, root):
    name = job["name"]
    stop_path = root / "STOP"
    if stop_path.exists() or time.time() >= job["deadline"]:
        raise BoundedStop(f"{name}: stop requested or protocol walltime exhausted")
    job_path = root / f"{name}.job.json"
    log_path = root / f"{name}.log"
    atomic_json(job_path, job)
    with log_path.open("a") as log:
        log_start = log.tell()
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--worker", str(job_path)], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(root / "active_worker.json", {
            "pid": process.pid, "job": name, "log": str(log_path),
            "progress": str(Path(job["checkpoint"]).parent / "progress.json") if job["kind"] == "train" else None,
            "started": time.time(), "deadline": job["deadline"], "status": "running"})
        try:
            while True:
                if stop_path.exists() or time.time() >= job["deadline"]:
                    raise BoundedStop(f"{name}: stop requested or protocol walltime exhausted")
                try:
                    code = process.wait(timeout=max(.001, min(1., job["deadline"] - time.time())))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (BoundedStop, KeyboardInterrupt):
            process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            atomic_json(root / "active_worker.json", {"pid": process.pid, "job": name,
                                                     "status": "stopped", "ended": time.time()})
            raise BoundedStop(f"{name}: supervisor stopped worker; see log")
    atomic_json(root / "active_worker.json", {"pid": process.pid, "job": name,
                                            "status": "complete" if code == 0 else "failed",
                                            "exit_code": code, "ended": time.time()})
    if code == 75:
        raise BoundedStop(f"{name}: bounded worker stop")
    if code:
        raise RuntimeError(f"{name}: worker exit {code}; see {root / (name + '.log')}")
    with log_path.open() as log:
        log.seek(log_start)
        if any("broadphase overflow" in line for line in log):
            raise RuntimeError(f"{name}: collision buffer overflow; evaluation/training invalid")


def validate_options(args):
    if not 1 <= args.max_updates <= 500 or not 1 <= args.eval_every <= args.max_updates:
        raise ValueError("updates must be within [1,500], evaluation interval <= budget")
    if not 0 < args.hours <= 24:
        raise ValueError("walltime must be within (0,24] hours")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("training seeds must be distinct")
    if set(args.seeds) & set(args.eval_seeds) or len(set(args.eval_seeds)) < 2:
        raise ValueError("use >=2 distinct heldout seeds disjoint from training")
    if (len(set(args.final_seeds)) < 2
            or set(args.final_seeds) & (set(args.seeds) | set(args.eval_seeds))):
        raise ValueError("final test seeds must be distinct from training and validation seeds")
    if args.trials < 128 or args.final_trials < 512 or args.eval_envs < 1:
        raise ValueError("validation requires >=128, final >=512 trials/shape/seed and positive env counts")
    if args.smoke_trials < 1:
        raise ValueError("smoke trials must be positive")
    if args.train_envs is not None and args.train_envs < 1:
        raise ValueError("training environment count must be positive")


def protocol_identity(args):
    return {key: value for key, value in vars(args).items()
            if key not in ("worker", "output", "report_only")}


def build_report(root, state):
    gate = hypothesis_gate(state["options"]["motion"], state["source_sha256"],
                           state["options"].get("source", str(SOURCE)),
                           state["options"].get("hypothesis", "small-motion"))
    results = state.get("checks", [])
    best = {}
    for seed in state["options"]["seeds"]:
        rows = [r for r in results if r["seed"] == seed and not r["assessment"]["regression"]]
        if rows:
            best[str(seed)] = max(rows, key=lambda r: r["recovery_success"])
    complete = state["status"] == "complete"
    final = state.get("final", [])
    promoted = (complete and gate["allowed"] and len(final) == len(state["options"]["seeds"])
                and all(item["assessment"]["certified"] for item in final))
    report = {
        "status": state["status"], "gate": gate, "promoted": promoted,
        "conclusion": ("validated_candidate" if promoted else
                       "skipped_failed_prerequisite" if not gate["allowed"] else
                       "failed" if state["status"] == "failed" else
                       "inconclusive" if complete else state["status"]),
        "best_matched_validation": best, "final": final, "error": state.get("error"),
        "error_history": state.get("error_history", []),
        "actual_updates": state.get("actual_updates", {}),
        "validation_seeds": state["options"].get("eval_seeds"),
        "final_test_seeds": state["options"].get("final_seeds"),
        "smoke_only": state.get("smoke_only", False),
        "conditional_steps": {
            name: {"status": "skipped" if not gate["allowed"] else "conditional",
                   "reason": gate["reason"] if not gate["allowed"] else "subject to bounded protocol"}
            for name in ("smoke", "recovery_curriculum", "matched_control_training", "final_candidate_evaluation")},
        "correlated_exploration": {"status": "unexecuted/deferred",
            "reason": "No tested likelihood-consistent correlated PPO distribution; "
                      "no post-action noise or unobserved scripted GRU burn-in implemented."},
        "limits": ["Recovery-designated resets are collision-free approaches, not proven learned contact recovery.",
                   "Bernoulli episode assignment has expected 75/25 mix; realized reset counts are checkpointed.",
                   "Worker boundaries restart episodes and both recurrent states; Adam and normalization continue.",
                   "Heldout cylinder/mesh pairs retain native solver contact limitations; inspect worker warnings.",
                   "Validation is exploratory, repeated looks and seed variation may be inconclusive.",
                   "No automatic deployment; promotion requires final guarded heldout evaluation for every seed."]}
    sensitivity = Path(state["options"]["motion"]).with_name("sensitivity.json")
    if sensitivity.is_file():
        evidence = read_json(sensitivity)
        report["threshold_sensitivity"] = {
            "path": str(sensitivity.resolve()), "sha256": digest(sensitivity),
            "caveat": evidence.get("caveat"),
            "mean_action_timeout_tail_2s": evidence.get("mean_actions", {}).get("timeout_tail_2s"),
            "role": "Supporting evidence only; never replaces or relaxes the actual curriculum gate."}
    atomic_json(root / "summary.json", report)
    return report


def supervise(args):
    validate_options(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    # flock is process-owned, so stale locks after interruption cannot block resumption.
    import fcntl
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = protocol_identity(args)
        state_path = root / "status.json"
        source_hash = digest(args.source)
        source = torch.load(args.source, map_location="cpu", weights_only=True)
        if source["iter"] != 19610:
            raise ValueError("this frozen experiment requires source iteration 19610")
        for role in ("actor", "critic"):
            model_cfg = source["infos"]["runner"][role]
            if (model_cfg["rnn_type"].lower() != "gru"
                    or model_cfg["rnn_hidden_dim"] != 128 or model_cfg["rnn_num_layers"] != 2):
                raise ValueError("source must use native two-layer GRU128")
        del source
        if state_path.exists():
            state = read_json(state_path)
            if state["options"] != identity or state["source_sha256"] != source_hash:
                raise ValueError("resume options/source differ from immutable protocol identity")
        else:
            state = {"version": VERSION, "options": identity, "source_sha256": source_hash,
                     "status": "waiting_for_motion_gate", "checks": [], "final": [],
                     "actual_updates": {}, "started": None, "stages": {},
                     "stopped_seeds": [], "selected": {}}

        def persist():
            state["updated"] = time.time()
            atomic_json(state_path, state)
            build_report(root, state)

        if args.report_only or state["status"] == "complete":
            if state["status"] == "complete":
                clear_stale_error(state)
            persist()
            return
        gate = hypothesis_gate(args.motion, source_hash, args.source, args.hypothesis)
        if "gate_identity" in state and state["gate_identity"] != gate:
            raise ValueError("hypothesis evidence changed; use a distinct output for a revised gate")
        state["gate_identity"] = gate
        state["gate"] = gate
        if not gate["allowed"]:
            state["status"] = "waiting_for_motion_gate"
            persist()
            return
        if state["started"] is None:
            state["started"] = time.time()
        deadline = state["started"] + args.hours * 3600
        common = {"source": args.source, "source_sha256": source_hash, "device": args.device,
                  "deadline": deadline, "train_envs": args.train_envs, "eval_envs": args.eval_envs,
                  "eval_seeds": args.eval_seeds, "hypothesis": args.hypothesis,
                  "final_seeds": args.final_seeds, "gate_identity": gate}

        def execute(name, kind, **kwargs):
            if (root / "STOP").exists() or time.time() >= deadline:
                raise BoundedStop("stop requested or protocol walltime exhausted (deadline persists across resume)")
            clear_stale_error(state)
            state.update(status="running", current_job=name)
            persist()
            run_job({**common, "name": name, "kind": kind, **kwargs}, root)

        def evaluation(name, checkpoint, trials, final=False):
            seeds = args.final_seeds if final else args.eval_seeds
            path = root / f"{name}.json"
            if path.exists():
                result = read_json(path)
                if (result["checkpoint_sha256"] != digest(checkpoint)
                        or result["trials_per_shape_per_seed"] != trials
                        or result["eval_seeds"] != seeds):
                    raise ValueError("cached evaluation identity mismatch")
                return result["metrics"]
            execute(name, "eval", checkpoint=str(checkpoint), trials=trials, output=str(path),
                    eval_seeds=seeds, evaluation_role="final_test" if final else "validation")
            return read_json(path)["metrics"]

        try:
            if (root / "STOP").exists():
                raise BoundedStop("STOP exists; remove this experiment's STOP file to resume")
            baseline = evaluation("frozen", args.source, args.smoke_trials if args.smoke else args.trials)
            if args.smoke:
                for arm in ("control", "recovery"):
                    folder = root / f"smoke_{arm}"
                    folder.mkdir(exist_ok=True)
                    execute(f"smoke_{arm}", "train", checkpoint=str(folder / "checkpoint.pt"),
                            seed=args.seeds[0], arm=arm, target=1, stage=0)
                    evaluation(f"smoke_{arm}_eval", folder / "checkpoint.pt", args.smoke_trials)
                    state["actual_updates"][f"{args.seeds[0]}/{arm}"] = 1
                state["status"] = "complete"
                clear_stale_error(state)
                state["smoke_only"] = True
                persist()
                return
            budgets = list(range(args.eval_every, args.max_updates + 1, args.eval_every))
            if args.max_updates not in budgets:
                budgets.append(args.max_updates)
            for target in budgets:
                for seed in args.seeds:
                    if seed in state["stopped_seeds"]:
                        continue
                    if any(r["seed"] == seed and r["updates"] == target for r in state["checks"]):
                        continue
                    stage = state["stages"].get(str(seed), 0)
                    pair = {}
                    for arm in ("control", "recovery"):
                        folder = root / f"seed_{seed}" / arm
                        folder.mkdir(parents=True, exist_ok=True)
                        checkpoint = folder / "checkpoint.pt"
                        execute(f"train_{seed}_{arm}_{target}", "train", checkpoint=str(checkpoint),
                                seed=seed, arm=arm, target=target, stage=stage)
                        progress = read_json(folder / "progress.json")
                        if progress["updates"] != target:
                            raise RuntimeError("training did not reach matched update budget")
                        state["actual_updates"][f"{seed}/{arm}"] = target
                        archive = folder / f"update_{target}.pt"
                        if not archive.exists():
                            # Atomic copy; retain matched candidates, never select by training reward.
                            atomic_checkpoint(archive, torch.load(checkpoint, map_location="cpu", weights_only=True))
                        pair[arm] = evaluation(f"eval_{seed}_{arm}_{target}", archive, args.trials)
                    assessment = assess(pair["recovery"], pair["control"], baseline, stage)
                    control_guard = assess(pair["control"], baseline, baseline, stage)
                    assessment["regression"] |= control_guard["regression"]
                    check = {"seed": seed, "updates": target, "stage": stage, "assessment": assessment,
                             "control_regression": control_guard["regression"],
                             "recovery_success": pair["recovery"][f"recovery/{stage}/all"]["success"]["rate"]}
                    state["checks"].append(check)
                    # Rank all candidates on the same stage-zero challenge, not easier-changing stages.
                    score = pair["recovery"]["recovery/0/all"]["success"]["rate"]
                    selected = state["selected"].get(str(seed))
                    if (not assessment["regression"] and assessment["promising"]
                            and (selected is None or score > selected["score"])):
                        state["selected"][str(seed)] = {"updates": target, "score": score, "stage": stage}
                    if assessment["certified"] and not assessment["regression"]:
                        state["stages"][str(seed)] = min(stage + 1, len(OFFSETS) - 1)
                    if (assessment["regression"]
                            or (target >= 300 and not any(r["seed"] == seed and r["assessment"]["promising"]
                                                         for r in state["checks"]))):
                        state["stopped_seeds"].append(seed)
                    persist()
            for seed in args.seeds:
                selected = state["selected"].get(str(seed))
                if selected is None or seed in state["stopped_seeds"]:
                    continue
                if any(item["seed"] == seed for item in state["final"]):
                    continue
                budget = selected["updates"]
                pair = {}
                for arm in ("control", "recovery"):
                    checkpoint = root / f"seed_{seed}" / arm / f"update_{budget}.pt"
                    pair[arm] = evaluation(f"final_{seed}_{arm}_{budget}", checkpoint, args.final_trials, final=True)
                frozen_final = evaluation("frozen_final", args.source, args.final_trials, final=True)
                state["final"].append({"seed": seed, "updates": budget,
                    "assessment": assess(pair["recovery"], pair["control"], frozen_final, selected["stage"])})
                persist()
            state["status"] = "complete"
            clear_stale_error(state)
        except (BoundedStop, KeyboardInterrupt) as exc:
            state.update(status="bounded_stop", error=str(exc))
        except BaseException:
            state.update(status="failed", error=traceback.format_exc())
            persist()
            raise
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--motion", default=str(MOTION))
    parser.add_argument("--hypothesis", choices=("small-motion", "oscillatory-recovery"),
                        default="small-motion")
    parser.add_argument("--output", default="runs/recovery_experiment_19610")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[31415, 27182])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[42, 2026])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[104729, 130363])
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--hours", type=float, default=18.)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--final-trials", type=int, default=512)
    parser.add_argument("--eval-envs", type=int, default=8)
    parser.add_argument("--train-envs", type=int, default=None,
                        help="default preserves checkpoint counts, including per-shape layout")
    parser.add_argument("--smoke", action="store_true",
                        help="separate output: frozen eval, one update per arm, post-update eval")
    parser.add_argument("--smoke-trials", type=int, default=4,
                        help="smoke only; deliberately insufficient for promotion")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    if args.worker:
        job = read_json(args.worker)
        try:
            if job["kind"] == "train":
                train_worker(job)
            elif job["kind"] == "eval":
                evaluate_worker(job)
            else:
                raise ValueError("unknown job kind")
        except BoundedStop:
            logging.exception("bounded stop")
            raise SystemExit(75)
        except BaseException:
            logging.exception("worker failed")
            raise
    else:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        supervise(args)


if __name__ == "__main__":
    main()
