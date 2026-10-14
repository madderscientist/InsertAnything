"""策略导出、检查点恢复、多卡分布式训练与可视化场景测试"""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import RNNModel

from export_policy import OfflinePolicy
from insertion import ALL_SHAPES, InsertionCfg, InsertionEnv, runner_config
from insertion.distributed import run_distributed, validate_layout
from train import (ContinuousRunner, TrainingStopped, atomic_json, contract,
                   health_state, load_recovery_checkpoint, prepare_runner_config)
from viser_view import SnapshotScene, replay_config, replay_index


class FakeScene:
    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        del kwargs
        return SimpleNamespace(name=name, vertices=vertices, faces=faces, remove=lambda: None)

    def add_frame(self, name, **kwargs):
        del kwargs
        return SimpleNamespace(name=name, remove=lambda: None)


class PolicyAndViewerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        cfg = prepare_runner_config(runner_config())["actor"]
        cfg.pop("class_name")
        obs = TensorDict({"actor": torch.zeros(2, 13)}, [2])
        self.actor = RNNModel(obs, {"actor": ["actor"]}, "actor", 6, **cfg).eval()

    @torch.inference_mode()
    def test_masked_reset_and_exported_guards(self):
        obs = TensorDict({"actor": torch.randn(2, 13)}, [2])
        self.actor(obs)
        state = self.actor.get_hidden_state()
        assert isinstance(state, torch.Tensor)
        before = state.clone()
        self.actor.reset(torch.tensor([True, False]))
        after = self.actor.get_hidden_state()
        assert isinstance(after, torch.Tensor)
        torch.testing.assert_close(after[:, 0], torch.zeros_like(after[:, 0]))
        torch.testing.assert_close(after[:, 1], before[:, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            torch.jit.script(self.actor.as_jit()).save(str(path / "actor.pt"))
            (path / "contract.json").write_text(json.dumps(contract(InsertionCfg())))
            policy = OfflinePolicy(path)
            p, q = torch.tensor([[0., 0, 0.005]]), torch.tensor([[1., 0, 0, 0]])
            goal, wrench = torch.tensor([[0., 0, -0.02]]), torch.zeros(1, 6)
            first = policy.step(p, q, goal, q, wrench)
            policy.reset()
            again = policy.step(p, q, goal, q, wrench)
            for a, b in zip(first, again, strict=True):
                torch.testing.assert_close(a, b)
            wrench[0, 0] = 31
            with self.assertRaisesRegex(RuntimeError, "force limit"):
                policy.step(p, q, goal, q, wrench)

    def test_viser_scene_and_replay_helpers(self):
        server = SimpleNamespace(scene=FakeScene(), atomic=nullcontext)
        cfg = InsertionCfg()
        for shape in ALL_SHAPES:
            scene = SnapshotScene(server, shape, cfg)
            grasp = np.array([0.00125, -0.002, 0.0005])
            plant = {"qpos": [0.001, -0.002, 0.06, 1., 0., 0., 0.],
                     "grasp_offset": grasp.tolist(), "goal": [0., 0., -0.015]}
            scene.update(plant)
            np.testing.assert_allclose(scene.tip.position,
                                       np.array(plant["qpos"][:3]) + scene.nominal_tip + grasp)
            scene.remove()
        zero = replay_config(asdict(cfg), "square", True)
        self.assertEqual(zero.force_noise, 0.)
        self.assertEqual(replay_index(1., 400), 50)


class RuntimeAndDistributedTests(unittest.TestCase):
    def test_watchdog_and_corrupt_checkpoint_recovery(self):
        self.assertEqual(health_state(None, 100, 10, None, 120), "running")
        self.assertEqual(health_state(None, 200, 10, None, 120), "stalled")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            saved = {"actor_state_dict": {}, "critic_state_dict": {},
                     "optimizer_state_dict": {}, "iter": 7, "infos": {}}
            torch.save(saved, output / "previous.pt")
            (output / "checkpoint.pt").write_bytes(b"truncated")
            with self.assertLogs(level="WARNING"):
                recovered = load_recovery_checkpoint(output, "cpu")
            self.assertEqual(recovered["iter"], 7)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_distributed_training_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        cfg = InsertionCfg(envs_per_shape=8, episode_seconds=0.08,
                           clearance_curriculum=True, noise_curriculum=True,
                           noise_curriculum_mode="success",
                           physics_cuda_graph=True, observation_cuda_graph=True)
        config = runner_config(4)
        config["save_interval"] = 1
        metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg),
                    "simulation_devices": ["cuda:0", "cuda:1"], "collection_backend": "distributed"}
        self.assertEqual(validate_layout(metadata, "cuda:0"), 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_json(output / "config.json", metadata)
            env = InsertionEnv(cfg, "cuda:0", ("cuda:0", "cuda:1"))
            runner = ContinuousRunner(env, config, output, metadata)
            runner.learn(1)
            del runner, env
            torch.cuda.empty_cache()
            run_distributed(output, "cuda:0", iterations=2, verify=True)
            saved = torch.load(output / "checkpoint.pt", weights_only=True)
            self.assertEqual(saved["iter"], 2)
            self.assertEqual(saved["infos"]["progress"], {"policy_steps": 12, "transitions": 480})
            restored = ContinuousRunner(InsertionEnv(cfg, "cuda:0"), config, output, metadata)
            restored.alg.load(saved, None, True)
            for name, value in restored.alg.actor.state_dict().items():
                torch.testing.assert_close(value, saved["actor_state_dict"][name])
            restored.stop_requested = True
            with self.assertRaises(TrainingStopped):
                restored.save("unused")


if __name__ == "__main__":
    unittest.main()
