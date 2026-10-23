"""导出确定性 TorchScript / ONNX 策略并验证序列数值等价性与推理延迟"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tensordict import TensorDict

from evaluate import load_checkpoint
from insertion import action_target, actor_observation, rotate, transform_wrench
from insertion.transforms import rotvec_quat


@torch.inference_mode()
def export_onnx(checkpoint: Path, output: Path, samples: int = 500) -> dict:
    import onnx
    import onnxruntime as ort

    if samples < 100:
        raise ValueError("at least 100 samples required for sequence and latency validation")
    metadata, actor = load_checkpoint(checkpoint, "cpu")
    actor.eval()
    model = actor.as_onnx().eval()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "actor.onnx"
    torch.onnx.export(
        model, model.get_dummy_inputs(), str(path), input_names=model.input_names,
        output_names=model.output_names, opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(17)
    hidden = np.zeros((2, 1, 128), dtype=np.float32)
    actor.reset()
    max_action_error = max_hidden_error = 0.
    elapsed = []
    test_observations, test_actions, test_states = [], [], []
    for index in range(samples):
        if index % 37 == 0:
            hidden.fill(0)
            actor.reset()
        position = torch.from_numpy(rng.uniform(-0.01, 0.01, (1, 3)).astype(np.float32))
        quaternion = rotvec_quat(torch.from_numpy(
            rng.uniform(-0.35, 0.35, (1, 3)).astype(np.float32)))
        goal_quaternion = torch.tensor([[1., 0., 0., 0.]])
        goal = torch.tensor([[0., 0., -0.015]])
        wrench = torch.from_numpy((rng.normal(size=(1, 6)) *
                                  [2., 2., 2., 0.05, 0.05, 0.05]).astype(np.float32))
        obs = actor_observation(position, quaternion, goal, goal_quaternion, wrench)
        expected = actor(TensorDict({"actor": obs}, [1])).numpy()
        expected_hidden = actor.get_hidden_state().numpy()
        start = time.perf_counter_ns()
        actions, hidden = session.run(None, {"obs": obs.numpy(), "h_in": hidden})
        elapsed.append((time.perf_counter_ns()-start)/1e6)
        np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(hidden, expected_hidden, rtol=1e-4, atol=1e-5)
        max_action_error = max(max_action_error, float(np.max(np.abs(actions-expected))))
        max_hidden_error = max(max_hidden_error, float(np.max(np.abs(hidden-expected_hidden))))
        test_observations.append(obs.numpy().copy())
        test_actions.append(expected.copy())
        test_states.append(expected_hidden.copy())
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = {
        **metadata["contract"],
        "onnx": {
            "inputs": {"obs": [1, 13], "h_in": [2, 1, 128]},
            "outputs": {"actions": [1, 6], "h_out": [2, 1, 128]},
            "dtype": "float32", "batch_size": 1, "deterministic": True,
            "normalization_in_model": True, "action_clipping_in_model": False,
            "preprocessing_in_model": False, "safety_checks_in_model": False,
        },
        "source": {
            "checkpoint": str(checkpoint.resolve()), "iteration": source["iter"],
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "curriculum_level": metadata.get("noise_curriculum_state", {}).get("level"),
            "environment": metadata["environment"],
        },
    }
    report = {
        "sequence_samples": samples, "resets_every": 37,
        "action_max_absolute_error": max_action_error,
        "hidden_max_absolute_error": max_hidden_error,
        "p50_ms": float(np.median(elapsed)), "p99_ms": float(np.percentile(elapsed, 99)),
        "includes_robot_io": False, "device": "cpu",
        "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "test_vectors.npz", obs=np.stack(test_observations),
        actions=np.stack(test_actions), h_out=np.stack(test_states),
        reset=(np.arange(samples) % 37 == 0))
    shutil.copy2(Path(__file__).with_name("DEPLOYMENT_ONNX.md"), output / "DEPLOYMENT_ONNX.md")
    return report


class OfflinePolicy:
    """No robot I/O. Caller supplies calibrated SI-unit state at 50 Hz."""

    def __init__(self, export_dir: Path):
        self.metadata = json.loads((export_dir / "contract.json").read_text())
        if self.metadata["version"] != 1:
            raise ValueError("unsupported exported contract")
        self.model = torch.jit.load(str(export_dir / "actor.pt"), map_location="cpu").eval()
        self.reset()

    def reset(self) -> None:
        self.model.reset()

    @torch.inference_mode()
    def step(self, tip_position: torch.Tensor, tool_quaternion: torch.Tensor,
             goal_position: torch.Tensor, goal_quaternion: torch.Tensor,
             tip_wrench: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = (tip_position, tool_quaternion, goal_position, goal_quaternion, tip_wrench)
        if any(v.shape != (1, width) for v, width in zip(values, (3, 4, 3, 4, 6), strict=True)):
            raise ValueError("offline policy requires batch-one tensors of widths 3,4,3,4,6")
        if any(v.device.type != "cpu" or v.dtype != torch.float32 or
               not torch.isfinite(v).all() for v in values):
            raise ValueError("offline state must be finite CPU float32")
        for q in (tool_quaternion, goal_quaternion):
            if not torch.allclose(q.norm(dim=-1), torch.ones(1), atol=1e-4):
                raise ValueError("offline quaternions must have unit length")
        if torch.linalg.vector_norm(tip_wrench[:, :3]) > self.metadata["force_limit_n"]:
            raise RuntimeError("offline force limit exceeded; caller must stop the robot")
        if torch.linalg.vector_norm(tip_wrench[:, 3:]) > self.metadata["moment_limit_nm"]:
            raise RuntimeError("offline moment limit exceeded; caller must stop the robot")
        observation = actor_observation(*values)
        action = self.model(observation)
        if not torch.isfinite(action).all():
            raise FloatingPointError("non-finite actor output")
        p, q = action_target(tip_position, tool_quaternion, action,
                             self.metadata["translation_limit_m"],
                             self.metadata["rotation_component_limit_rad"])
        lateral = (torch.linalg.vector_norm(p[:, :2], dim=-1)
                   if self.metadata.get("workspace_xy_geometry") == "radial_disk"
                   else p[:, :2].abs())
        if (lateral > self.metadata["workspace_xy_m"]).any() or (
            p[:, 2] < self.metadata["workspace_z_low_m"]).any() or (
            p[:, 2] > self.metadata["workspace_z_high_m"]).any():
            raise RuntimeError("offline target leaves configured workspace")
        axis = rotate(q, torch.tensor([[0., 0, 1.]]))
        if bool((axis[:, 2] < math.cos(self.metadata["max_tilt_rad"])).any()):
            raise RuntimeError("offline target tilt exceeds configured limit")
        return p, q


@torch.inference_mode()
def export(checkpoint: Path, output: Path, samples: int = 500) -> dict:
    if samples < 100:
        raise ValueError("at least 100 samples required for latency measurements")
    metadata, actor = load_checkpoint(checkpoint, "cpu")
    output.mkdir(parents=True, exist_ok=False)
    actor.reset()
    exported = torch.jit.script(actor.as_jit().eval())
    exported.reset()
    generator = torch.Generator().manual_seed(17)
    max_error = 0.0
    for index in range(100):
        obs = torch.randn(1, 13, generator=generator)
        original = actor(TensorDict({"actor": obs}, [1]))
        actual = exported(obs)
        torch.testing.assert_close(original, actual, rtol=1e-5, atol=1e-6)
        max_error = max(max_error, float((original-actual).abs().max()))
        if index % 19 == 0:
            actor.reset()
            exported.reset()
    exported.save(str(output / "actor.pt"))
    (output / "contract.json").write_text(json.dumps(metadata["contract"], indent=2) + "\n")
    policy = OfflinePolicy(output)
    position = torch.tensor([[0., 0, 0.005]])
    quaternion = torch.tensor([[1., 0, 0, 0]])
    goal = torch.tensor([[0., 0, -0.02]])
    force = torch.tensor([[0., 0, 1.]])
    moment = torch.zeros(1, 3)
    sensor_to_tip = torch.tensor([[0., 0, -0.08]])
    elapsed = []
    for i in range(samples + 30):
        start = time.perf_counter_ns()
        wrench = transform_wrench(force, moment, quaternion, sensor_to_tip)
        policy.step(position, quaternion, goal, quaternion, wrench)
        milliseconds = (time.perf_counter_ns() - start) / 1e6
        if i >= 30:
            elapsed.append(milliseconds)
    p99 = float(np.percentile(elapsed, 99))
    report = {"sequence_max_absolute_error": max_error, "device": "cpu",
              "batch_size": 1, "samples": samples, "p50_ms": float(np.median(elapsed)),
              "p99_ms": p99, "overruns_20ms": sum(x >= 20 for x in elapsed),
              "meets_50hz_p99": p99 < 20, "includes_robot_io": False}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if p99 >= 20:
        raise RuntimeError(f"offline p99 {p99:.2f} ms exceeds the 20 ms budget")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/export"))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--onnx", action="store_true",
                        help="export deterministic batch-one ONNX with explicit GRU state")
    args = parser.parse_args()
    torch.set_num_threads(1)
    exporter = export_onnx if args.onnx else export
    print(json.dumps(exporter(args.checkpoint, args.output, args.samples), indent=2))


if __name__ == "__main__":
    main()
