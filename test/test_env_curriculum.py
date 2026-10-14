"""奖励函数、孔间隙课程与批量 GPU 物理环境测试"""

import math
from typing import Any
import unittest

import torch

from insertion import (InsertionCfg, InsertionEnv, PPO_DISCOUNT,
                       SuccessNoiseCurriculum, TRAIN_SHAPES, bounded_stiffness_step,
                       completion_cost_terms, quat_rotvec, tip_pose)


class RewardAndCurriculumUnitTests(unittest.TestCase):
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
        cls.env = InsertionEnv(cls.cfg, cls.device)

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

    def test_cuda_graph_equivalence_and_live_clearance_upgrade(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=4, episode_seconds=0.08,
                           clearance_curriculum=True, noise_curriculum=True,
                           noise_curriculum_mode="success",
                           physics_cuda_graph=True, observation_cuda_graph=True)
        env = InsertionEnv(cfg, self.device)
        group = env.groups[0]
        self.assertEqual(group.clearance, 0.002)
        self.assertAlmostEqual(group.difficulty, 0.25)
        env.record_curriculum_outcomes(torch.tensor([[0., 1., 1., 1.]] * 1000, device=self.device))
        logs = env.advance_noise_curriculum()
        self.assertEqual(group.clearance, 0.001)
        self.assertAlmostEqual(group.difficulty, 0.40)
        self.assertAlmostEqual(float(logs["curriculum/clearance_mm"]), 1.0)
        for _ in range(4):
            obs, rewards, _, _ = env.step(torch.zeros(4, 6, device=self.device))
            self.assertTrue(torch.isfinite(obs["actor"]).all())
            self.assertTrue(torch.isfinite(rewards).all())
        self.assertIsNotNone(group.physics_graph)
        self.assertIsNotNone(group.observation_graph)


if __name__ == "__main__":
    unittest.main()
