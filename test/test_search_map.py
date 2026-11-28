"""空间概率记忆 C1：先验校准、只减不加的擦除、一步 NLL 公式与端到端训练"""

from argparse import Namespace
from dataclasses import asdict, replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
import torch

import play
from evaluate import MapPolicy, load_checkpoint
from insertion import InsertionCfg, InsertionEnv
from insertion.distributed import run_distributed
from insertion.search_map import (CELL_M, DENSITY_MAX, DENSITY_MIN, DENSITY_SCALE, ERASE_MAX, GRID, LOG_FLOOR,
                                  MAP_COARSE, MAP_FINE, MAP_NLL, PATCH, SearchMap, erase_nll, prior_log_mass)
from real2sim import evaluate_real2sim
from search_prior.train_search import search_runner
from train import ContinuousRunner, atomic_json, contract, restore_environment_progress
from viser_view import replay_config

SEARCH = dict(position_bias=0.005, initial_xy=0.005, workspace_xy_radius_m=0.025)


class SearchMapTests(unittest.TestCase):
    def test_zero_noise_map_replay_keeps_a_normalized_prior(self):
        cfg = InsertionCfg(shapes=("triangle",), search_map=True, **SEARCH)
        replay = replay_config(asdict(cfg), "triangle", zero_noise=True)
        self.assertEqual(replay.position_bias, 0.)
        self.assertTrue(replay.search_map)
        with self.assertRaisesRegex(ValueError, "position_bias"):
            replace(replay, training=True)
        with self.assertRaisesRegex(ValueError, "position_bias"):
            replace(replay, position_bias=0.016)
        env = Namespace(device="cpu", search_map=SearchMap(1, "cpu"),
                        groups=[Namespace(cfg=replay, episode_noise_scale=torch.ones(1))])
        InsertionEnv.reset_search_map(env, torch.ones(1, dtype=torch.bool))
        torch.testing.assert_close(env.search_map.log_prior, prior_log_mass(torch.zeros(1)))
        torch.testing.assert_close(env.search_map.log_prior.exp().sum(), torch.tensor(1.))

    def test_replay_entry_points_bind_their_own_map_before_first_action(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=1, search_map=True, **SEARCH)
        actor = Mock(spec=MapPolicy)
        for entry in ("play", "real2sim"):
            with self.subTest(entry=entry):
                memory = SearchMap(1, "cpu")
                env = Mock(num_envs=1, max_episode_length=1, groups=[Namespace(n=1, shape="square")],
                           search_map=memory)
                action = torch.zeros(1, 6)
                terminal = {"reason": torch.ones(1), "length": torch.ones(1)}
                def step(actions):
                    self.assertIs(actor.search_map, memory)
                    return env.get_observations(), torch.zeros(1), torch.ones(1, dtype=torch.bool), {"terminal": terminal}
                env.step.side_effect = step
                env.get_observations.return_value = {"actor": torch.zeros(1, 13)}
                actor.return_value = action
                if entry == "play":
                    with patch.object(play, "InsertionEnv", return_value=env), \
                         patch.object(play, "load_checkpoint", return_value=({"environment": asdict(cfg)}, actor)), \
                         patch("sys.argv", ["play", "checkpoint.pt", "--headless", "--steps", "1", "--device", "cpu"]):
                        play.main()
                else:
                    flags = torch.zeros(1, dtype=torch.bool)
                    with patch.object(evaluate_real2sim, "InsertionEnv", return_value=env), \
                         patch.object(evaluate_real2sim, "probe", return_value=(torch.zeros(1, 3), flags, flags)):
                        rows = evaluate_real2sim.evaluate_real(actor, cfg, "cpu", 1)
                    self.assertEqual(len(rows), 1)
                env.step.assert_called_once_with(action)

    def test_episode_prior_sigma_is_log_uniform_and_independent_of_noise(self):
        cfg = InsertionCfg(search_map=True, search_map_sigma_min_m=0.0005, search_map_sigma_max_m=0.003)
        group = Namespace(cfg=cfg, episode_noise_scale=torch.zeros(2000))
        env = Namespace(search_map=SearchMap(2000, "cpu"), groups=[group], device="cpu",
                        map_generator=torch.Generator().manual_seed(0))
        InsertionEnv.reset_search_map(env, torch.ones(2000, dtype=torch.bool))
        p = env.search_map.log_prior.exp()
        axis = (torch.arange(GRID) - GRID / 2 + 0.5) * CELL_M
        std = (p.sum(2) * axis.square()).sum(1).sqrt()
        self.assertGreater(float(std.min()), 0.0004)
        self.assertLess(float(std.max()), 0.0031)
        log_std = std.log()
        self.assertAlmostEqual(float(log_std.mean()), (math.log(0.0005) + math.log(0.003)) / 2, delta=0.1)
        with self.assertRaises(ValueError):
            InsertionCfg(search_map_sigma_min_m=0.0005, search_map_sigma_max_m=0.003)

    def test_resume_builds_map_prior_from_the_reset_episodes(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=1, search_map=True, noise_curriculum=True,
                           grasp_position_error=0., tip_calibration_error=0., **SEARCH)
        group = Namespace(cfg=cfg, n=1, device="cpu", episode_noise_scale=torch.zeros(1),
                          generator=torch.Generator(), walk_generator=torch.Generator())
        group.reset = lambda ids: group.episode_noise_scale.index_fill_(0, ids, group.noise_scale)
        env = InsertionEnv.__new__(InsertionEnv)
        env.device, env.num_envs, env.groups, env.noise_curriculum = "cpu", 1, [group], None
        env.search_map = SearchMap(1, "cpu")
        env.reset_search_map(torch.ones(1, dtype=torch.bool))
        startup_prior = env.search_map.log_prior.clone()

        saved = {"infos": {"progress": {"policy_steps": cfg.noise_full_steps}, "environment": asdict(cfg)}}
        restore_environment_progress(env, cfg, saved, 1)

        self.assertEqual(float(group.episode_noise_scale[0]), 1.)
        expected = prior_log_mass(torch.tensor([cfg.position_bias / math.sqrt(3)]))
        self.assertFalse(torch.allclose(startup_prior, expected))
        torch.testing.assert_close(env.search_map.log_prior, expected)

    def test_prior_is_normalized_gaussian_with_given_std(self):
        std = torch.tensor([0.001, 0.003, 0.005])
        p = prior_log_mass(std).exp()
        torch.testing.assert_close(p.sum((1, 2)), torch.ones(3))
        centers = (torch.arange(GRID) - GRID / 2 + 0.5) * CELL_M
        # 离散化后方差多出格宽平方的 1/12
        variance = (p.sum(2) * centers.square()).sum(1)
        torch.testing.assert_close(variance, std.square() + CELL_M**2 / 12, rtol=0.02, atol=0)

    def test_erase_only_clamps_and_stays_in_patch(self):
        n = 3
        memory = SearchMap(n, "cpu")
        prior = prior_log_mass(torch.full((n,), 0.004))
        memory.reset(torch.arange(n), prior)
        tip = torch.tensor([[0., 0.], [0.0235, -0.0235], [0.05, 0.]])
        obs = memory.observe(tip, torch.zeros(n, 2))
        self.assertEqual(obs[MAP_FINE].shape, (n, 2, 9, 9))
        # 角落处的窗口：图内为全图 log-softmax 密度，图外为零质量填充
        density = ((torch.log_softmax(prior[1].flatten(), -1).view(GRID, GRID) + math.log(GRID * GRID))
                   .clamp(DENSITY_MIN, DENSITY_MAX) / DENSITY_SCALE)
        expected = torch.full((9, 9), DENSITY_MIN / DENSITY_SCALE)
        expected[:5, 4:] = density[GRID - 5:, :5]
        torch.testing.assert_close(obs[MAP_FINE][1, 0].float(), expected, atol=2e-3, rtol=0)
        self.assertEqual(obs[MAP_COARSE].shape, (n, 3, GRID // 4, GRID // 4))
        self.assertEqual(obs[MAP_FINE].dtype, torch.float16)
        for _ in range(20):
            memory.pending = torch.full((n, PATCH * PATCH), ERASE_MAX)
            memory.apply()
        self.assertTrue(bool((memory.erased <= 0).all()))
        self.assertAlmostEqual(float(memory.erased.min()), -LOG_FLOOR, places=5)
        self.assertEqual(int((memory.erased < 0).sum((1, 2))[0]), PATCH * PATCH)
        rows, cols, _ = memory.cache
        self.assertEqual(rows[1].tolist(), list(range(GRID - PATCH, GRID)))
        self.assertEqual(cols[1].tolist(), list(range(PATCH)))
        self.assertEqual(memory.visits.sum((1, 2)).tolist(), [20., 20., 20.])
        memory.reset(torch.tensor([0]), prior[:1])
        self.assertEqual(float(memory.erased[0].abs().sum()), 0.)
        self.assertLess(float(memory.erased[1].sum()), 0.)

    def test_removed_mass_tracks_soft_erasure(self):
        n = 2
        memory = SearchMap(n, "cpu")
        prior = prior_log_mass(torch.full((n,), 0.002))
        memory.reset(torch.arange(n), prior)
        generator = torch.Generator().manual_seed(0)
        for _ in range(40):
            memory.observe((torch.rand(n, 2, generator=generator) - 0.5) * 0.01, torch.zeros(n, 2))
            memory.pending = torch.rand(n, PATCH * PATCH, generator=generator) * ERASE_MAX
            memory.apply()
        expected = (prior.exp() * -torch.expm1(memory.erased)).sum((1, 2))
        torch.testing.assert_close(memory.removed, expected, atol=1e-5, rtol=0)
        self.assertTrue(bool((memory.removed > 0).all()))
        memory.reset(torch.tensor([0]), prior[:1])
        self.assertEqual(float(memory.removed[0]), 0.)

    def test_nll_formula_matches_full_map_update(self):
        torch.manual_seed(0)
        n = 64
        memory = SearchMap(n, "cpu")
        prior = prior_log_mass(torch.rand(n) * 0.007 + 0.001)
        memory.reset(torch.arange(n), prior)
        memory.erased.copy_(-torch.rand(n, GRID, GRID) * LOG_FLOOR)
        tip = (torch.rand(n, 2) - 0.5) * 0.05
        truth = tip + (torch.rand(n, 2) - 0.5) * 0.008
        obs = memory.observe(tip, truth)
        logits = torch.randn(n, PATCH * PATCH) * 3
        after, before = erase_nll(logits, obs[MAP_NLL])
        memory.pending = ERASE_MAX * torch.sigmoid(logits)
        memory.apply()
        log_p = torch.log_softmax((memory.log_prior + memory.erased).flatten(1), -1).view(n, GRID, GRID)
        cell = (truth / CELL_M + GRID / 2).floor().long().clamp(0, GRID - 1)
        expected = -log_p[torch.arange(n), cell[:, 0], cell[:, 1]]
        torch.testing.assert_close(after, expected, atol=1e-4, rtol=1e-4)
        self.assertTrue(bool((before > 0).all()))

    def test_env_slices_match_unpadded_recurrent_batches(self):
        # 更新时地图观测按环境切片直接取，须与 rsl_rl 填充后再去填充的结果逐一相同
        torch.manual_seed(0)
        steps, envs, chunks = 6, 8, 4
        obs = TensorDict({"x": torch.zeros(envs, 3)}, [envs])
        storage = RolloutStorage("rl", envs, steps, obs, [6], "cpu")
        storage.observations["x"].copy_(torch.randn(steps, envs, 3))
        storage.dones.copy_(torch.rand(steps, envs, 1) < 0.3)
        storage.distribution_params = (torch.zeros(steps, envs, 6),)
        size = envs // chunks
        for index, batch in enumerate(storage.recurrent_mini_batch_generator(chunks, 2)):
            envs_slice = slice(index % chunks * size, (index % chunks + 1) * size)
            torch.testing.assert_close(unpad_trajectories(batch.observations["x"], batch.masks),
                                       storage.observations["x"][:, envs_slice], rtol=0, atol=0)

    def test_map_policy_trains_and_resumes_with_both_heads(self):
        torch.set_num_threads(1)
        # 回合长于一次 rollout，结束时地图尚未随回合重置
        cfg = InsertionCfg(envs_per_shape=4, shapes=("square",), episode_seconds=0.2, search_map=True,
                           search_coverage_reward=5.0, **SEARCH)
        config = search_runner(Namespace(steps=4, rnn_hidden=64, rnn_layers=None, mini_batches=2,
                                         gsde_resample_steps=25, phase_coef=0.1, phase_lr=3e-4,
                                         search_map=True, map_coef=0.5, map_lr=1e-3))
        metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_json(output / "config.json", metadata)
            env = InsertionEnv(cfg, "cuda:0")
            runner = ContinuousRunner(env, config, output, metadata)
            self.assertIs(runner.alg.search_map, env.search_map)
            self.assertTrue(hasattr(runner.alg.actor, "map_encoder"))
            self.assertFalse(hasattr(runner.alg.critic, "map_encoder"))
            head = runner.alg.critic.map_head
            before = head.net[-1].weight.detach().clone()
            phase_before = runner.alg.critic.phase_head.weight.detach().clone()
            runner.learn(1)
            self.assertFalse(torch.equal(before, head.net[-1].weight.detach()))
            self.assertFalse(torch.equal(phase_before, runner.alg.critic.phase_head.weight))
            self.assertEqual([g["lr"] for g in runner.alg.optimizer.param_groups[1:]],
                             [runner.alg.learning_rate, 3e-4, 1e-3])
            self.assertTrue(bool((env.search_map.erased < 0).any()))
            self.assertTrue(math.isfinite(float(env.search_map.metrics["map/info_gain"])))
            saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertIn("net.0.weight", {k.removeprefix("map_head.") for k in saved["critic_state_dict"]})
            with self.assertRaises(NotImplementedError):
                runner.alg.actor.as_jit()
            self.assertIsInstance(load_checkpoint(output / "checkpoint.pt", "cuda:0")[1], MapPolicy)
            resumed = ContinuousRunner(env, config, output, metadata)
            resumed.alg.load(saved, None, True)
            torch.testing.assert_close(resumed.alg.critic.phase_head.weight, runner.alg.critic.phase_head.weight)
            torch.testing.assert_close(resumed.alg.critic.map_head.net[-1].weight, head.net[-1].weight)
            self.assertEqual([g["lr"] for g in resumed.alg.optimizer.param_groups],
                             [runner.alg.learning_rate, runner.alg.learning_rate, 3e-4, 1e-3])
            resumed.learn(1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
    def test_distributed_map_training_keeps_ranks_in_sync(self):
        torch.set_num_threads(1)
        cfg = InsertionCfg(envs_per_shape=8, shapes=("square", "triangle"), episode_seconds=0.2,
                           search_map=True, physics_cuda_graph=True, observation_cuda_graph=True, **SEARCH)
        config = search_runner(Namespace(steps=4, rnn_hidden=64, rnn_layers=None, mini_batches=2,
                                         gsde_resample_steps=25, phase_coef=0.1, phase_lr=1e-3,
                                         search_map=True, map_coef=0.5, map_lr=1e-3))
        metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg),
                    "simulation_devices": ["cuda:0", "cuda:1"], "collection_backend": "distributed"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_json(output / "config.json", metadata)
            run_distributed(output, "cuda:0", iterations=2, verify=True)
            saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(saved["iter"], 1)
            self.assertTrue(any(key.startswith("map_encoder.") for key in saved["actor_state_dict"]))
            self.assertIn("phase_head.weight", saved["critic_state_dict"])
            self.assertIn("map_head.net.0.weight", saved["critic_state_dict"])


if __name__ == "__main__":
    unittest.main()
