"""奖励函数、孔间隙课程与批量 GPU 物理环境测试"""

import math
from dataclasses import replace
from itertools import product
from typing import Any
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import torch

from insertion import (InsertionCfg, InsertionEnv, PPO_DISCOUNT,
                       SuccessNoiseCurriculum, TRAIN_SHAPES, bounded_stiffness_step,
                       completion_cost_terms, quat_rotvec, tip_pose)
from insertion.curriculum import adaptive_environment_counts
from insertion.transforms import rotvec_quat
from insertion.env import PlantGroup


class RewardAndCurriculumUnitTests(unittest.TestCase):
    def test_depth_success_hold_reset_safety_and_legacy_alignment(self):
        cfg = InsertionCfg()
        self.assertFalse(cfg.success_alignment_required)
        self.assertTrue(InsertionCfg.from_checkpoint({}).success_alignment_required)
        group = SimpleNamespace(
            cfg=cfg, geometry=SimpleNamespace(peg_length=0.035, radius=0.01, depth=0.025),
            clearance=0.0001, offset=torch.zeros(1, 3),
            goal=torch.tensor([[0., 0., -0.015]]),
            hold=torch.zeros(1, dtype=torch.long), length=torch.zeros(1, dtype=torch.long),
            peak_force=torch.zeros(1), peak_moment=torch.zeros(1),
            wrench=torch.zeros(1, 6), last_action=torch.zeros(1, 6),
            alignment=lambda q: (torch.tensor([0.01]), torch.tensor([0.02])),
        )
        position = torch.tensor([[0.0001, 0., -0.015]])
        quaternion = torch.tensor([[1., 0., 0., 0.]])
        group.state = lambda: (position, quaternion, torch.zeros(1, 3), torch.zeros(1, 3))
        actions = torch.zeros(1, 6)
        for _ in range(4):
            _, done, _ = PlantGroup.step_outcome(group, actions)
            self.assertFalse(done.item())
        _, done, info = PlantGroup.step_outcome(group, actions)
        self.assertTrue(done.item())
        self.assertEqual(info["reason"].item(), 1)
        self.assertEqual(group.last_reward_terms["completion_bonus"].item(), 20)
        position[:, 2] = -0.0147
        PlantGroup.step_outcome(group, actions)
        self.assertEqual(group.hold.item(), 0)
        position[:, 2] = -0.01481
        PlantGroup.step_outcome(group, actions)
        self.assertEqual(group.hold.item(), 1)
        group.cfg = replace(cfg, success_alignment_required=True)
        PlantGroup.step_outcome(group, actions)
        self.assertEqual(group.hold.item(), 0)
        group.cfg = cfg
        for field, value, reason in (("peak_force", 31., 2), ("peak_moment", 2.1, 2)):
            group.hold.fill_(4)
            getattr(group, field).fill_(value)
            _, done, info = PlantGroup.step_outcome(group, actions)
            self.assertTrue(done.item())
            self.assertEqual(info["reason"].item(), reason)
            self.assertEqual(group.last_reward_terms["completion_bonus"].item(), 0)
            getattr(group, field).zero_()
        group.hold.fill_(4)
        position[:, 0] = 0.026
        _, _, info = PlantGroup.step_outcome(group, actions)
        self.assertEqual(info["reason"].item(), 3)

    def test_adaptive_counts_preserve_total_mixture_and_device_shards(self):
        for unit in (1, 2, 4, 8):
            counts = (512,) * 5
            self.assertEqual(adaptive_environment_counts(counts, [0.] * 5, unit), counts)
            self.assertEqual(adaptive_environment_counts(counts, [0.2] * 5, unit), counts)
            result = adaptive_environment_counts(counts, [1., 0., 0., 0., 0.], unit)
            self.assertEqual(sum(result), 2560)
            self.assertTrue(all(n % unit == 0 and n >= unit for n in result))
            targets = [0.3*512 + 0.7*2560] + [0.3*512]*4
            self.assertTrue(all(abs(n-t) <= unit for n, t in zip(result, targets)))
            self.assertGreater(result[0], 768)
            self.assertLess(result[1], 256)
            self.assertEqual(adaptive_environment_counts(result, [0.] * 5, unit), counts)
            mixed = adaptive_environment_counts(counts, [0.1, 0.2, 0.3, 0.4, 0.5], unit)
            targets = [0.3*512 + 0.7*2560*f/1.5 for f in (0.1, 0.2, 0.3, 0.4, 0.5)]
            self.assertEqual(sum(mixed), 2560)
            self.assertTrue(all(abs(n-t) <= unit for n, t in zip(mixed, targets)))
        self.assertEqual(adaptive_environment_counts((1, 1, 1), [1., 0., 0.]), (1, 1, 1))
        for kwargs in ({"shape_env_counts": (1,)}, {"force_bias": -1},
                       {"moment_bias": float("nan")}, {"adaptive_sampling": True},
                       {"initial_z_min": 0.}, {"initial_z_min": 0.02},
                       {"initial_z_max": float("nan")}, {"initial_z_max": 0.03}):
            with self.assertRaises(ValueError):
                InsertionCfg(**kwargs)
        old = InsertionCfg.from_checkpoint({"envs_per_shape": 512})
        self.assertEqual(old.force_bias, 0.)
        self.assertEqual(old.environment_counts, (512,) * 5)
        self.assertEqual((old.initial_z_min, old.initial_z_max), (0.004, 0.01))

    def terms(self, z=0.001, xy=0., depth=0.015, tilt=0., yaw=0., force=0.,
              moment=0., success=False, failure=False, timeout=False, action=0., previous=0.):
        cfg = InsertionCfg()
        scalar = lambda x: torch.tensor([x], dtype=torch.float64)
        return completion_cost_terms(
            cfg, scalar(xy), scalar(z + depth), scalar(depth), 0.01,
            scalar(math.radians(tilt)), scalar(math.radians(yaw)),
            torch.tensor([[0., 0., force, 0., 0., moment]], dtype=torch.float64),
            torch.full((1, 6), action, dtype=torch.float64),
            torch.full((1, 6), previous, dtype=torch.float64),
            torch.tensor([success]), torch.tensor([failure]), torch.tensor([timeout]))

    def reward(self, **values):
        return float(sum(self.terms(**values).values()))

    def test_quadratic_load_cost_and_retired_checkpoint_field(self):
        self.assertAlmostEqual(float(self.terms(force=20.)["force_penalty"]), -0.0072)
        self.assertAlmostEqual(float(self.terms(force=17.5)["force_penalty"]),
                               0.25 * float(self.terms(force=30.)["force_penalty"]))
        self.assertAlmostEqual(float(self.terms(moment=1.05)["moment_penalty"]), -0.005)
        legacy = InsertionCfg.from_checkpoint({"load_cost_power": 2})
        self.assertFalse(hasattr(legacy, "load_cost_power"))
        with self.assertRaisesRegex(ValueError, "explicit reward migration"):
            InsertionCfg.from_checkpoint({"load_cost_power": 1})

    def test_reward_shaping_mouth_gradient_and_safety_bound(self):
        cfg = InsertionCfg()
        hover = self.reward()
        self.assertLess(hover, 0)
        hover_return = hover * (1 - PPO_DISCOUNT ** 400) / (1 - PPO_DISCOUNT)
        complete_return = sum(PPO_DISCOUNT ** i * self.reward(z=-0.015, success=(i == 4))
                              for i in range(5))
        self.assertGreater(complete_return, hover_return)
        # 孔口表面 0.2 mm 的平面修正收益必须显著大于动作成本
        mouth_delta = (float(self.terms(z=0., xy=0.)["position_penalty"])
                       - float(self.terms(z=0., xy=0.0002)["position_penalty"]))
        self.assertGreater(mouth_delta, 0.002)
        self.assertLess(self.reward(z=0.001, tilt=5.), hover)
        self.assertLess(self.reward(z=0.001, yaw=5.), self.reward(z=0.001, tilt=5.))
        self.assertLess(float(self.terms(z=-0.013, tilt=3.)["orientation_penalty"]),
                        float(self.terms(z=-0.002, tilt=3.)["orientation_penalty"]))
        self.assertEqual(float(self.terms(force=5.)["force_penalty"]), 0)
        self.assertLess(float(self.terms(force=6.)["force_penalty"]), 0)
        worst = self.reward(z=1000., xy=1000., tilt=180., yaw=180.,
                            force=1000., moment=1000., action=100., previous=-100.)
        self.assertGreaterEqual(worst, -cfg.completion_step_cost_bound)
        self.assertLess(self.reward(failure=True), -cfg.completion_step_cost_bound / (1 - PPO_DISCOUNT))

    def test_bounded_stiffness_walk_math(self):
        initial = torch.tensor([[300., 1500., 2500., 4., 8., 12.]], dtype=torch.float64)
        lower, upper = initial * 0.9, initial * 1.1
        current = initial.clone()
        for index in range(200):
            before = current.clone()
            target = upper if index < 100 else lower
            current = bounded_stiffness_step(current, target, initial, lower, upper, 0.01, 1., 0.05)
            self.assertTrue(bool((current >= lower).all() & (current <= upper).all()))
            self.assertTrue(bool(((current - before).abs() <= initial * 0.05 * 0.01 + 1e-10).all()))

    def test_clearance_difficulty_and_noise_curriculum(self):
        levels = (0.002, 0.001, 0.0005, 0.00025, 0.00015, 0.0001)
        curriculum = SuccessNoiseCurriculum(("square", "triangle"), clearances=levels, noise_levels=10)
        self.assertEqual(curriculum.clearance, 0.002)
        self.assertAlmostEqual(curriculum.difficulty, 0.25)
        self.assertEqual(curriculum.scale, 0.)
        self.assertFalse(curriculum.update({"square": [True] * 999, "triangle": [True] * 1000}))
        for i, expected_clearance in enumerate(levels[1:], start=1):
            self.assertTrue(curriculum.update({"square": [True] * 1000, "triangle": [True] * 1000}))
            self.assertEqual(curriculum.clearance, expected_clearance)
            self.assertAlmostEqual(curriculum.difficulty, 0.25 + 0.75 * (i / 5))
            self.assertEqual(curriculum.scale, 0.)
        self.assertTrue(curriculum.update({"square": [True] * 1000, "triangle": [True] * 1000}))
        self.assertEqual(curriculum.clearance, 0.0001)
        self.assertAlmostEqual(curriculum.difficulty, 1.0)
        self.assertAlmostEqual(curriculum.scale, 0.1)
        restored = SuccessNoiseCurriculum(("square", "triangle"), clearances=levels, noise_levels=10)
        restored.load_state_dict(curriculum.state_dict())
        self.assertEqual(restored.state_dict(), curriculum.state_dict())
        invalid_cases: tuple[dict[str, Any], ...] = (
            {"time_cost_per_second": 0}, {"safety_failure_penalty": 1},
            {"noise_curriculum_mode": "success"},
        )
        for values in invalid_cases:
            with self.assertRaises(ValueError):
                InsertionCfg(**values)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class BatchedEnvIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = "cuda:0"
        cls.cfg = InsertionCfg(
            shapes=TRAIN_SHAPES, envs_per_shape=2, training=False, dynamics="nominal",
            initial_xy=0, initial_tilt=0, initial_yaw=0, position_bias=0,
            tip_calibration_error=0, grasp_position_error=0, sensor_rotation_error=0,
            position_noise=0, rotation_noise=0, force_noise=0, moment_noise=0,
        )
        cls.env = InsertionEnv(replace(cls.cfg, physics_cuda_graph=True,
                                      observation_cuda_graph=True), cls.device)

    def test_scripted_aligned_insertion_succeeds_all_shapes(self):
        for group in self.env.groups:
            group.reset(torch.arange(group.n, device=self.device))
        reached = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        with torch.inference_mode():
            for _ in range(250):
                pairs = []
                for group in self.env.groups:
                    pos, q, _, _ = group.state()
                    tip = tip_pose(pos, q, group.offset)
                    error = group.goal - tip
                    angle = quat_rotvec(q)
                    pairs.append(torch.cat((error / self.cfg.translation_limit,
                                            -angle / self.cfg.rotation_limit), -1).clamp(-1, 1))
                _, _, dones, info = self.env.step(torch.cat(pairs))
                reached |= dones & (info["terminal"]["reason"] == 1)
                if reached.all():
                    break
        self.assertTrue(bool(reached.all()))

    def test_episode_wrench_bias_and_unequal_shape_layout(self):
        cfg = replace(self.cfg, shapes=("triangle", "square"), envs_per_shape=3,
                      shape_env_counts=(2, 4), force_bias=0.5, moment_bias=0.01,
                      physics_cuda_graph=True, observation_cuda_graph=True)
        env = InsertionEnv(cfg, self.device)
        self.assertEqual([g.n for g in env.groups], [2, 4])
        self.assertEqual(env.num_envs, 6)
        group = env.groups[0]
        group.noise_scale = 0.5
        group.reset(torch.arange(group.n, device=self.device))
        before = group.wrench_bias.clone()
        self.assertTrue(bool((before[:, :3].abs() <= 0.25).all()))
        self.assertTrue(bool((before[:, 3:].abs() <= 0.005).all()))
        self.assertGreater(float(before.abs().sum()), 0.)
        actor, critic = group.observations()
        torch.testing.assert_close(actor[:, -6:], group.wrench + before)
        zeros = torch.zeros_like(before)
        clean_actor, clean_critic = group.observations_from_noise(
            zeros, zeros[:, :3], zeros[:, :3])
        group.wrench_bias.zero_()
        unbiased_actor, unbiased_critic = group.observations_from_noise(
            zeros, zeros[:, :3], zeros[:, :3])
        torch.testing.assert_close(clean_actor[:, -6:]-unbiased_actor[:, -6:], before)
        torch.testing.assert_close(clean_critic, unbiased_critic)
        group.wrench_bias.copy_(before)
        env.step(torch.zeros(6, 6, device=self.device))
        torch.testing.assert_close(group.wrench_bias, before)
        group.reset(torch.tensor([0], device=self.device))
        torch.testing.assert_close(group.wrench_bias[1], before[1])
        self.assertFalse(torch.equal(group.wrench_bias[0], before[0]))
        group.noise_scale = 0.
        group.reset(torch.arange(group.n, device=self.device))
        torch.testing.assert_close(group.wrench_bias, torch.zeros_like(before))

    def test_expanded_initial_true_tip_and_rotation_vector_ranges(self):
        cfg = replace(self.cfg, shapes=("square",), envs_per_shape=512,
                      initial_xy=0.01, initial_z_min=0.001, initial_z_max=0.01,
                      initial_tilt=math.radians(20), initial_yaw=math.radians(20))
        env = InsertionEnv(cfg, self.device)
        group = env.groups[0]
        for difficulty in (1., 0.25):
            group.difficulty = difficulty
            group.reset(torch.arange(group.n, device=self.device))
            pos, q, _, _ = group.state()
            tip = tip_pose(pos, q, group.offset)
            rv = quat_rotvec(q)
            for axis in (0, 1):
                values = tip[:, axis]
                bound = 0.01 * difficulty
                self.assertTrue(bool((values.abs() <= bound + 1e-7).all()))
                self.assertLess(float(values.min()), -0.9 * bound)
                self.assertGreater(float(values.max()), 0.9 * bound)
            self.assertTrue(bool(((tip[:, 2] >= 0.001-1e-7) & (tip[:, 2] <= 0.01+1e-7)).all()))
            self.assertLess(float(tip[:, 2].min()), 0.0019)
            self.assertGreater(float(tip[:, 2].max()), 0.0091)
            for axis in range(3):
                bound = math.radians(20) * difficulty
                self.assertTrue(bool((rv[:, axis].abs() <= bound + 1e-6).all()))
                self.assertLess(float(rv[:, axis].min()), -0.9 * bound)
                self.assertGreater(float(rv[:, axis].max()), 0.9 * bound)

    def test_initial_plane_clearance_all_shapes_and_grasp_offsets(self):
        cfg = replace(self.cfg, shapes=TRAIN_SHAPES+("circle",), envs_per_shape=64,
                      initial_xy=0.01, initial_z_min=0.001, initial_z_max=0.01,
                      initial_plane_margin=0.001, initial_tilt=math.radians(20),
                      initial_yaw=math.radians(20), grasp_position_error=0.00125)
        env = InsertionEnv(cfg, self.device)
        for group in env.groups:
            pos, q, _, _ = group.state()
            tip = tip_pose(pos, q, group.offset)
            minimum = group.initial_minimum_tip_height(q)
            self.assertTrue(bool((tip[:, 2] >= minimum-1e-7).all()))
            self.assertTrue(bool((tip[:, 2] <= 0.01+1e-7).all()))
            before = group.sim.data.qpos.clone()
            group.reset(torch.tensor([0], device=self.device))
            torch.testing.assert_close(group.sim.data.qpos[1:], before[1:])
            # Independent MuJoCo forward validates the actual collision model, including grasp offsets.
            for i in range(8):
                grasp = (group.offset[i]-group.nominal_offset[i]).cpu().numpy()
                for geom_id in group.peg_geom_ids:
                    group.model.geom_pos[geom_id] = group.default_geom_pos[geom_id].cpu().numpy()+grasp
                data = mujoco.MjData(group.model)
                data.qpos[:] = group.sim.data.qpos[i].cpu().numpy()
                mujoco.mj_forward(group.model, data)
                self.assertFalse(any(data.contact[j].dist < 0 for j in range(data.ncon)))
            tilted = rotvec_quat(torch.tensor(
                [[math.radians(20), math.radians(20), math.radians(20)]], device=self.device))
            self.assertLess(float(group.initial_minimum_tip_height(tilted)[0]), cfg.initial_z_max)
        group = env.groups[0]
        with patch.object(group, "initial_minimum_tip_height",
                          return_value=torch.full((1,), 0.02, device=self.device)):
            with self.assertRaisesRegex(RuntimeError, "16 attempts"):
                group.reset(torch.tensor([0], device=self.device))
        self.assertEqual(InsertionCfg.from_checkpoint({}).initial_plane_margin, 0.)
        for value in (-1., float("nan"), 0.02):
            with self.assertRaises(ValueError):
                replace(cfg, initial_plane_margin=value)

    def test_cuda_graph_equivalence_and_live_clearance_upgrade(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=4, episode_seconds=0.08,
                           clearance_curriculum=True, noise_curriculum=True,
                           noise_curriculum_mode="success",
                           physics_cuda_graph=True, observation_cuda_graph=True)
        env = InsertionEnv(cfg, self.device)
        group = env.groups[0]
        self.assertEqual(group.clearance, 0.002)
        self.assertAlmostEqual(group.difficulty, 0.25)
        env.step(torch.zeros(4, 6, device=self.device))
        self.assertIsNotNone(group.physics_rollout_graph)
        self.assertIsNotNone(group.reset_constants_graph)
        self.assertIsNotNone(group.action_graph)
        self.assertIsNotNone(group.outcome_graph)
        env.record_curriculum_outcomes(torch.tensor([[0., 1., 1., 1.]] * 1000, device=self.device))
        logs = env.advance_noise_curriculum()
        self.assertEqual(group.clearance, 0.001)
        self.assertIsNone(group.physics_rollout_graph)
        self.assertIsNone(group.reset_constants_graph)
        self.assertIsNone(group.action_graph)
        self.assertIsNone(group.outcome_graph)
        self.assertAlmostEqual(group.difficulty, 0.40)
        torch.testing.assert_close(group.peg_polygon, torch.tensor(
            group.geometry.peg_polygon, dtype=torch.float32, device=self.device))
        self.assertAlmostEqual(float(logs["curriculum/clearance_mm"]), 1.0)
        for _ in range(4):
            obs, rewards, _, _ = env.step(torch.zeros(4, 6, device=self.device))
            self.assertTrue(torch.isfinite(obs["actor"]).all())
            self.assertTrue(torch.isfinite(rewards).all())
        self.assertIsNotNone(group.physics_graph)
        self.assertIsNotNone(group.physics_rollout_graph)
        self.assertIsNotNone(group.reset_constants_graph)
        self.assertIsNotNone(group.action_graph)
        self.assertIsNotNone(group.outcome_graph)
        self.assertIsNotNone(group.observation_graph)

    def test_full_physics_graph_matches_eager_with_contacts_and_resets(self):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        for device, dynamics in product(devices, ("nominal", "train")):
            with self.subTest(device=device, dynamics=dynamics):
                cfg = replace(self.cfg, envs_per_shape=2, episode_seconds=0.4,
                              dynamics=dynamics)
                eager = InsertionEnv(cfg, device)
                captured = InsertionEnv(replace(cfg, physics_cuda_graph=True,
                                               observation_cuda_graph=True), device)
                actions = torch.zeros(eager.num_envs, 6, device=device)
                actions[:, 2] = -1
                saw_contact = False
                saw_reset = False
                with torch.inference_mode():
                    for step in range(24):
                        actions[:, 0] = 0.05 * math.sin(step)
                        actions[:, 5] = 0.02 * math.cos(step)
                        obs, reward, done, info = eager.step(actions)
                        other_obs, other_reward, other_done, other_info = captured.step(actions)
                        torch.testing.assert_close(other_obs, obs, rtol=1e-5, atol=2e-6)
                        torch.testing.assert_close(other_reward, reward, rtol=1e-5, atol=2e-6)
                        torch.testing.assert_close(other_done, done)
                        for key in info["terminal"]:
                            torch.testing.assert_close(other_info["terminal"][key],
                                                       info["terminal"][key], rtol=1e-5, atol=2e-6)
                        for left, right in zip(eager.groups, captured.groups, strict=True):
                            for name in ("qpos", "qvel", "time"):
                                torch.testing.assert_close(getattr(right.sim.data, name)[:],
                                                           getattr(left.sim.data, name)[:],
                                                           rtol=1e-5, atol=2e-6)
                            for name in ("wrench", "raw_wrench", "k", "d", "hold", "length"):
                                torch.testing.assert_close(getattr(right, name), getattr(left, name),
                                                           rtol=1e-5, atol=2e-6)
                            for key in left.last_reward_terms:
                                torch.testing.assert_close(right.last_reward_terms[key],
                                                           left.last_reward_terms[key],
                                                           rtol=1e-5, atol=2e-6)
                            saw_contact |= bool((left.peak_force > 0.1).any())
                            self.assertIsNotNone(right.physics_rollout_graph)
                        saw_reset |= bool(done.any())
                self.assertTrue(saw_contact)
                self.assertTrue(saw_reset)

    def test_step_graph_outputs_are_owned_and_invalid_actions_still_raise(self):
        cfg = replace(self.cfg, shapes=("square",), physics_cuda_graph=True,
                      observation_cuda_graph=True)
        env = InsertionEnv(cfg, self.device)
        group = env.groups[0]
        actions = torch.zeros(env.num_envs, 6, device=self.device)
        group.step(actions)
        reward, done, terminal = group.step(actions)
        saved_reward, saved_done = reward.clone(), done.clone()
        saved_terminal = {key: value.clone() for key, value in terminal.items()}
        actions[:, 2] = -1
        group.step(actions)
        torch.testing.assert_close(reward, saved_reward)
        torch.testing.assert_close(done, saved_done)
        for key, value in terminal.items():
            torch.testing.assert_close(value, saved_terminal[key])
        actions[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite policy action"):
            env.step(actions)
        actions.zero_()
        group.sim.data.qpos[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "invalid physical state"):
            group.step(actions)


if __name__ == "__main__":
    unittest.main()
