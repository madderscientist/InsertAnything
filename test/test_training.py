"""策略导出、检查点恢复、多卡分布式训练与可视化场景测试"""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import RNNModel

from export_policy import OfflinePolicy
from insertion import ALL_SHAPES, InsertionCfg, InsertionEnv, SuccessNoiseCurriculum, runner_config
from insertion.distributed import run_distributed, validate_layout
from train import (ContinuousRunner, TrainingStopped, atomic_json, contract,
                   adapt_training_layout, environment_config, health_state, load_recovery_checkpoint,
                   main, prepare_runner_config, restore_environment_progress, resume_training)
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
        cfg = InsertionCfg(force_bias=0.5, moment_bias=0.01,
                           shape_env_counts=(15, 17, 16, 16, 16))
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
        self.assertEqual(zero.force_bias, 0.)
        self.assertEqual(zero.moment_bias, 0.)
        self.assertEqual(zero.environment_counts, (1,))
        self.assertEqual(replay_index(1., 400), 50)


class RuntimeAndDistributedTests(unittest.TestCase):
    def test_finite_and_continuous_share_environment_config(self):
        argv = ["train.py", "--initial-xy", "0.01", "--initial-z-min", "0.001",
                "--initial-tilt-deg", "20", "--initial-yaw-deg", "20",
                "--force-bias", "0.5", "--moment-bias", "0.01",
                "--success-noise-curriculum", "--steps", "16"]
        configs = []
        for entry, extra in (("supervise", []), ("run_finite", ["--iterations", "1"])):
            with (patch("sys.argv", argv + extra),
                  patch(f"train.{entry}", side_effect=lambda args:
                        configs.append(environment_config(args)))):
                main()
        self.assertEqual(configs[0], configs[1])
        cfg = configs[0]
        self.assertEqual(cfg.initial_xy, 0.01)
        self.assertEqual(cfg.initial_z_min, 0.001)
        self.assertAlmostEqual(cfg.initial_tilt, np.deg2rad(20))
        self.assertEqual(cfg.initial_tilt, cfg.initial_yaw)
        self.assertEqual((cfg.force_bias, cfg.moment_bias), (0.5, 0.01))
        self.assertEqual((cfg.noise_zero_steps, cfg.noise_full_steps), (1600, 12800))
        self.assertTrue(cfg.noise_curriculum)
        self.assertEqual(cfg.noise_curriculum_mode, "success")

    def test_sampler_handoff_cooldown_smoothing_and_curriculum_change(self):
        cfg = InsertionCfg(envs_per_shape=512, noise_curriculum=True,
                           noise_curriculum_mode="success", adaptive_sampling=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = {"environment": asdict(cfg), "simulation_devices": ["cuda:0", "cuda:1"]}
            atomic_json(output / "config.json", metadata)
            curriculum = {"level": 4, "windows": {
                s: [True] * n + [False] * (1000-n)
                for s, n in zip(cfg.shapes, (600, 800, 800, 800, 800), strict=True)}}
            saved = {"actor_state_dict": {}, "critic_state_dict": {},
                     "optimizer_state_dict": {}, "iter": 10,
                     "infos": {"noise_curriculum_state": curriculum}}
            torch.save(saved, output / "checkpoint.pt")
            with (patch("train.registered_worker_pid", return_value=123),
                  patch("train.os.kill") as kill):
                self.assertFalse(adapt_training_layout(output, 10))
                self.assertFalse(adapt_training_layout(output, 109))
                with self.assertLogs(level="WARNING"):
                    self.assertTrue(adapt_training_layout(output, 110))
                kill.assert_called_once()
                updated = json.loads((output / "config.json").read_text())
                counts = updated["environment"]["shape_env_counts"]
                self.assertEqual(sum(counts), 2560)
                self.assertTrue(all(n % 2 == 0 for n in counts))
                self.assertGreater(counts[0], counts[1])
                self.assertTrue((output / "sampling_110.json").exists())
                self.assertFalse(adapt_training_layout(output, 110))
                curriculum["level"] = 5
                for values in curriculum["windows"].values():
                    values.clear()
                torch.save(saved, output / "checkpoint.pt")
                self.assertFalse(adapt_training_layout(output, 210))
                state = json.loads((output / "config.json").read_text())["adaptive_sampling_state"]
                self.assertEqual(state, {"level": 5, "iteration": 210})
                self.assertFalse(adapt_training_layout(output, 310))
                state = json.loads((output / "config.json").read_text())["adaptive_sampling_state"]
                self.assertEqual(state["iteration"], 310)
                self.assertEqual(kill.call_count, 1)

    def test_changed_sensor_bias_retains_level_but_invalidates_old_windows(self):
        cfg = InsertionCfg(noise_curriculum=True, noise_curriculum_mode="success",
                           force_bias=0.5, moment_bias=0.01)
        curriculum = SuccessNoiseCurriculum(cfg.shapes)
        old = {"level": 4, "windows": {s: [True] * 999 for s in cfg.shapes}}
        env = SimpleNamespace(noise_curriculum=curriculum, curriculum_pending=[1], groups=[])
        saved = {"infos": {"progress": {"policy_steps": 20},
                          "environment": {**asdict(cfg), "force_bias": 0., "moment_bias": 0.},
                          "noise_curriculum_state": old}}
        del saved["infos"]["environment"]["force_bias"]
        del saved["infos"]["environment"]["moment_bias"]
        with self.assertLogs(level="WARNING"):
            restore_environment_progress(env, cfg, saved, 0)
        self.assertEqual(curriculum.level, 4)
        self.assertTrue(all(not w for w in curriculum.windows.values()))
        self.assertEqual(env.curriculum_pending, [])
        saved["infos"]["environment"] = asdict(cfg)
        restore_environment_progress(env, cfg, saved, 1)
        self.assertTrue(all(len(w) == 999 for w in curriculum.windows.values()))
        for key, value in (("initial_xy", 0.01), ("initial_tilt", 0.35),
                           ("initial_yaw", 0.35), ("initial_z_min", 0.001),
                           ("initial_z_max", 0.02)):
            changed = InsertionCfg.from_checkpoint({**asdict(cfg), key: value})
            with self.assertLogs(level="WARNING"):
                restore_environment_progress(env, changed, saved, 2)
            self.assertEqual(curriculum.level, 4)
            self.assertTrue(all(not w for w in curriculum.windows.values()))
        del saved["infos"]["environment"]["initial_z_min"]
        del saved["infos"]["environment"]["initial_z_max"]
        restore_environment_progress(env, cfg, saved, 3)
        self.assertTrue(all(len(w) == 999 for w in curriculum.windows.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_biased_nonuniform_ppo_checkpoint_resume(self):
        torch.set_num_threads(1)
        cfg = InsertionCfg(envs_per_shape=4, shape_env_counts=(6, 4, 4, 4, 2),
                           episode_seconds=0.08, force_bias=0.5, moment_bias=0.01,
                           clearance_curriculum=True, noise_curriculum=True,
                           noise_curriculum_mode="success", adaptive_sampling=True,
                           physics_cuda_graph=True, observation_cuda_graph=True)
        config = runner_config(4)
        config["save_interval"] = 1
        metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_json(output / "config.json", metadata)
            runner = ContinuousRunner(InsertionEnv(cfg, "cuda:0"), config, output, metadata)
            runner.env.noise_curriculum.level = 14
            runner.learn(1)
            saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertTrue(saved["optimizer_state_dict"]["state"])
            resumed = ContinuousRunner(InsertionEnv(cfg, "cuda:0"), config, output, metadata)
            resume_training(resumed, cfg, 1)
            self.assertEqual(resumed.env.noise_curriculum.level, 14)
            self.assertEqual(resumed.env.num_envs, 20)
            self.assertTrue(all(bool(g.wrench_bias.abs().sum()) for g in resumed.env.groups))
            for name, value in resumed.alg.actor.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["actor_state_dict"][name])
            resumed.learn(1)
            heartbeat = json.loads((output / "heartbeat.json").read_text())
            self.assertEqual(list(heartbeat["shape_environment_counts"].values()), [6, 4, 4, 4, 2])
            self.assertTrue(heartbeat["adaptive_sampling"])

    def test_nonuniform_distributed_layout_validation_without_second_gpu(self):
        cfg = InsertionCfg(envs_per_shape=8, shape_env_counts=(12, 8, 8, 8, 4))
        metadata = {"environment": asdict(cfg), "runner": runner_config(4),
                    "simulation_devices": ["cuda:0", "cuda:1"]}
        with patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(validate_layout(metadata, "cuda:0"), 2)
            metadata["environment"]["shape_env_counts"] = [11, 9, 8, 8, 4]
            with self.assertRaises(ValueError):
                validate_layout(metadata, "cuda:0")

    def test_update_cache_released_once_per_simulation_device(self):
        runner = ContinuousRunner.__new__(ContinuousRunner)
        runner.env = SimpleNamespace(groups=[
            SimpleNamespace(device=device) for device in ("cuda:0", "cuda:0", "cuda:1")])
        with (patch("torch.cuda.device", side_effect=lambda _: nullcontext()),
              patch("torch.cuda.memory_allocated", return_value=100),
              patch("torch.cuda.memory_reserved", side_effect=[900, 150, 800, 140]),
              patch("torch.cuda.mem_get_info", return_value=(700, 1000)),
              patch("torch.cuda.empty_cache") as release):
            memory = runner.release_update_cache()
        self.assertEqual(release.call_count, 2)
        self.assertEqual(set(memory), {"cuda:0", "cuda:1"})
        self.assertEqual(memory["cuda:0"]["reserved_before_bytes"], 900)
        self.assertEqual(memory["cuda:0"]["reserved_after_bytes"], 150)
        self.assertEqual(memory["cuda:1"]["reserved_before_bytes"], 800)
        self.assertEqual(memory["cuda:1"]["reserved_after_bytes"], 140)

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
