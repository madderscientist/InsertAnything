"""real2sim 续训的定时健康巡检：存活/进度/崩溃统计、实机档评估、最佳 checkpoint 保存与退化回滚

用法（仓库根目录，后台常驻）:
    nohup setsid python -m real2sim.monitor --run runs/real2sim/stage5_contact_quiet &
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from train import atomic_json

SUCCESS_LINE = re.compile(r"success/(\w+): ([0-9.]+)")
ITERATION_LINE = re.compile(r"Learning iteration (\d+)/")
LOW_TRAIN_SUCCESS = 0.25
ROLLBACK_DROP = 0.15
ROLLBACK_COOLDOWN_SECONDS = 6 * 3600
SMOOTH_LINE = re.compile(r"Mean smooth loss: ([0-9.]+)")
# 连续 PROGRESS_WINDOW 次巡检无实质进展则自动调参；调参后至少再观察同样次数
PROGRESS_WINDOW = 3
SUCCESS_PROGRESS = 0.02
FLIP_PROGRESS = 0.005
DEGRADE_DROP = 0.05
# 开低通实机档平均成功用时上限；stage8 约 3.8 s，平滑正则过强会使策略过度保守
SLOW_SECONDS = 7.0
# 两档成功率与悬空反号率均达标后视为收敛，平台期不再自动调参
CONVERGED_SUCCESS = 0.97
CONVERGED_LOWPASS_FLIP = 0.03
CONVERGED_NO_LOWPASS_FLIP = 0.06
SMOOTH_COEF_RANGE = (0.0125, 0.4)
DESIRED_KL_MAX = 0.05
LR_FLOOR = 1.01e-5


def alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def log_tail_stats(log: Path, offset: int) -> tuple[dict, int]:
    with log.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode(errors="replace")
        end = handle.tell()
    iterations = [int(m) for m in ITERATION_LINE.findall(text)]
    blocks = text.split("Learning iteration")[-10:]
    rates = [float(v) for block in blocks for _, v in SUCCESS_LINE.findall(block)]
    smooth = [float(v) for v in SMOOTH_LINE.findall(text)][-10:]
    return {
        "recent_smooth_loss": sum(smooth) / len(smooth) if smooth else None,
        "iterations_seen": len(iterations),
        "last_iteration": iterations[-1] if iterations else None,
        "worker_crashes": text.count("FloatingPointError: invalid physical state"),
        "isolated_divergences": text.count("MuJoCo divergence"),
        "nan_reward": "Mean reward: nan" in text,
        "recent_train_success": sum(rates) / len(rates) if rates else None,
    }, end


def launch(run: Path) -> int:
    process = subprocess.Popen(
        [sys.executable, "-m", "real2sim.train_real2sim", "--output", str(run)],
        stdout=(run / "supervisor.out").open("a"), stderr=subprocess.STDOUT,
        start_new_session=True, cwd=Path(__file__).resolve().parents[1])
    return process.pid


def stop_supervisor(run: Path, timeout: float = 900.) -> None:
    (run / "STOP").touch()
    pid = read_json(run / "processes.json").get("supervisor_pid")
    deadline = time.time() + timeout
    while alive(pid) and time.time() < deadline:
        time.sleep(10)
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
        time.sleep(30)
    (run / "STOP").unlink(missing_ok=True)


def evaluate(run: Path, snapshot: Path, iteration: int, device: str, lowpass_off: bool = False) -> dict | None:
    output = run / "monitor_eval" / f"iter_{iteration:06d}{'_nolp' if lowpass_off else ''}"
    result = subprocess.run(
        [sys.executable, "-m", "real2sim.evaluate_real2sim", f"current={snapshot}",
         "--trials", "32", "--envs-per-shape", "32", "--device", device, "--output", str(output)]
        + (["--lowpass-off"] if lowpass_off else []),
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    summary = output / "summary.json"
    if result.returncode or not summary.exists():
        logging.error("Evaluation failed: %s", result.stderr[-2000:])
        return None
    return json.loads(summary.read_text())["checkpoints"]["current"]["overall"]


def retune(run: Path, changes: dict, rollback: bool) -> int:
    stop_supervisor(run)
    config = read_json(run / "config.json")
    config["runner"]["algorithm"].update(changes)
    atomic_json(run / "config.json", config)
    if rollback:
        shutil.copyfile(run / "checkpoint.pt", run / f"rolled_back_{int(time.time())}.pt")
        shutil.copyfile(run / "best_real_checkpoint.pt", run / "checkpoint.pt")
        (run / "previous.pt").unlink(missing_ok=True)
    return launch(run)


def track_progress(run: Path, state: dict, report: dict, metrics: dict, heartbeat: dict) -> str:
    entry = {"iteration": report["iteration"],
             "success": (metrics["success_rate"] + metrics["no_lowpass_success_rate"]) / 2,
             "flip": metrics["no_lowpass_free_air_sign_flip_ratio"] or 1.,
             "lr": heartbeat.get("learning_rate"), "smooth": report.get("recent_smooth_loss")}
    history = state.setdefault("history", [])
    history.append(entry)
    del history[:-12]
    state["check_count"] = state.get("check_count", 0) + 1
    best_success = (state["best"]["success_rate"] + state["best"]["no_lowpass_success_rate"]) / 2
    if entry["success"] < best_success - DEGRADE_DROP:
        verdict = "degraded"
    elif (metrics.get("mean_success_seconds") or 0.) > SLOW_SECONDS:
        verdict = "slow"
    elif (entry["success"] >= CONVERGED_SUCCESS and entry["flip"] <= CONVERGED_NO_LOWPASS_FLIP
          and (metrics.get("free_air_sign_flip_ratio") or 1.) <= CONVERGED_LOWPASS_FLIP):
        return "converged"
    elif len(history) <= PROGRESS_WINDOW:
        return "warming_up"
    else:
        anchor = history[-1 - PROGRESS_WINDOW]
        improved = (entry["success"] >= anchor["success"] + SUCCESS_PROGRESS
                    or entry["flip"] <= anchor["flip"] - FLIP_PROGRESS)
        verdict = "improving" if improved else "stagnant"
    checks_since = state["check_count"] - state.get("last_retune_index", -PROGRESS_WINDOW)
    cooldown = 2 if verdict in ("degraded", "slow") else PROGRESS_WINDOW
    if verdict == "improving" or checks_since < cooldown or (run / "STOP").exists():
        return verdict
    algorithm = read_json(run / "config.json")["runner"]["algorithm"]
    coef = algorithm.get("smooth_coef", 0.)
    changes: dict = {}
    if verdict == "degraded":
        changes["smooth_coef"] = max(SMOOTH_COEF_RANGE[0], coef / 2)
    elif verdict == "slow":
        changes["smooth_coef"] = max(SMOOTH_COEF_RANGE[0], coef / 1.6)
    else:
        if coef > 0:
            changes["smooth_coef"] = min(SMOOTH_COEF_RANGE[1], coef * 2)
        if all(h["lr"] is not None and h["lr"] <= LR_FLOOR for h in history[-PROGRESS_WINDOW:]):
            changes["desired_kl"] = min(DESIRED_KL_MAX, algorithm.get("desired_kl", 0.01) * 1.5)
    changes = {k: v for k, v in changes.items() if v != algorithm.get(k)}
    if changes:
        report["actions"].append({"retune": changes, "rollback": verdict == "degraded",
                                  "relaunch": retune(run, changes, verdict == "degraded")})
        state["last_retune_index"] = state["check_count"]
    return verdict


def check(run: Path, state: dict, device: str) -> dict:
    report: dict = {"timestamp": time.time(), "actions": []}
    processes = read_json(run / "processes.json")
    heartbeat = read_json(run / "heartbeat.json")
    report["iteration"] = heartbeat.get("iteration")
    report["heartbeat_age_s"] = time.time() - heartbeat["timestamp"] if heartbeat else None
    stats, state["log_offset"] = log_tail_stats(run / "training.log", state.get("log_offset", 0))
    report.update(stats)

    supervisor_alive = alive(processes.get("supervisor_pid"))
    stalled = report["iteration"] is not None and report["iteration"] == state.get("last_iteration")
    if not (run / "STOP").exists() and (not supervisor_alive or (stalled and state.get("stalled_before"))):
        if supervisor_alive:
            stop_supervisor(run)
        report["actions"].append({"relaunch": launch(run), "reason": "dead" if not supervisor_alive else "stalled"})
    state["stalled_before"] = stalled
    state["last_iteration"] = report["iteration"]

    checkpoint = run / "checkpoint.pt"
    if checkpoint.exists() and report["iteration"] is not None:
        snapshot = run / "monitor_snapshot.pt"
        shutil.copyfile(checkpoint, snapshot)
        metrics = evaluate(run, snapshot, report["iteration"], device)
        no_lowpass = evaluate(run, snapshot, report["iteration"], device, lowpass_off=True)
        report["real_profile"] = metrics
        report["real_profile_no_lowpass"] = no_lowpass
        best = state.get("best")
        if metrics is not None and no_lowpass is not None:
            metrics = {**metrics, "no_lowpass_success_rate": no_lowpass["success_rate"],
                       "no_lowpass_free_air_sign_flip_ratio": no_lowpass["free_air_sign_flip_ratio"]}
            score = lambda m: (round((m["success_rate"] + m.get("no_lowpass_success_rate", 0.)) / 2, 2),
                               -(m.get("no_lowpass_free_air_sign_flip_ratio") or 1.))
            if best is None or score(metrics) > score(best):
                shutil.copyfile(snapshot, run / "best_real_checkpoint.pt")
                state["best"] = {**metrics, "iteration": report["iteration"]}
                report["actions"].append({"new_best": report["iteration"]})
            report["verdict"] = track_progress(run, state, report, metrics, heartbeat)

    best = state.get("best")
    low = report["recent_train_success"] is not None and report["recent_train_success"] < LOW_TRAIN_SUCCESS
    state["low_checks"] = state.get("low_checks", 0) + 1 if (low or report["nan_reward"]) else 0
    metrics = report.get("real_profile")
    degraded = (best is not None and metrics is not None
                and metrics["success_rate"] < best["success_rate"] - ROLLBACK_DROP)
    if ((state["low_checks"] >= 3 or report["nan_reward"]) and degraded
            and time.time() - state.get("last_rollback", 0) > ROLLBACK_COOLDOWN_SECONDS):
        stop_supervisor(run)
        shutil.copyfile(run / "checkpoint.pt", run / f"rolled_back_from_{report['iteration']}.pt")
        shutil.copyfile(run / "best_real_checkpoint.pt", run / "checkpoint.pt")
        (run / "previous.pt").unlink(missing_ok=True)
        report["actions"].append({"rollback_to": best["iteration"], "relaunch": launch(run)})
        state["last_rollback"] = time.time()
        state["low_checks"] = 0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3600.)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = args.run.resolve()
    state_path = run / "monitor_state.json"
    state = read_json(state_path)
    while True:
        try:
            report = check(run, state, args.device)
        except Exception as error:  # 巡检本身异常不能终止常驻进程
            logging.exception("Health check failed")
            report = {"timestamp": time.time(), "error": repr(error)}
        with (run / "monitor.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        atomic_json(state_path, state)
        logging.info("%s", json.dumps(report))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
