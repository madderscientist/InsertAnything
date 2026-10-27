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
                       completion_cost_terms, geometry, quat_rotvec, tip_pose)
from insertion.curriculum import adaptive_environment_counts
from insertion.transforms import rotvec_quat
from insertion.env import PlantGroup, play_hysteresis_step


class RewardAndCurriculumUnitTests(unittest.TestCase):
    def test_curriculum_waits_for_full_cohort_including_slow_failures(self):
        curriculum = SuccessNoiseCurriculum(("square",))
        env = SimpleNamespace(
            noise_curriculum=curriculum, curriculum_pending=[], groups=[],
            curriculum_cohort=None, curriculum_cohort_seen=None)
        batch = torch.zeros(1000, 4)
        batch[:, 3] = 1
        batch[:799, 1:3] = 1
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(len(curriculum.windows["square"]), 0)
        # Repeated successes from fast environments must not displace slow failures.
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(len(curriculum.windows["square"]), 0)
        batch[:, 1:3] = 0
        batch[799:, 1] = 1
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(len(curriculum.windows["square"]), 1000)
        self.assertEqual(sum(curriculum.windows["square"]), 799)
        self.assertEqual(curriculum.level, 0)
        self.assertFalse(env.curriculum_cohort_seen.any())
        batch[:, 1:3] = 1
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(curriculum.level, 1)
        self.assertEqual(len(curriculum.windows["square"]), 0)

    def test_depth_success_hold_reset_safety_and_legacy_alignment(self):
        cfg = InsertionCfg(success_opening_required=False)
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
            minimum_peg_height_offset=lambda q: torch.zeros(q.shape[0]),
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
        group.cfg = replace(cfg, training=False, terminate_on_overload=False)
        group.peak_force.fill_(100)
        group.peak_moment.fill_(10)
        group.hold.fill_(4)
        _, done, info = PlantGroup.step_outcome(group, actions)
        self.assertTrue(done.item())
        self.assertEqual(info["reason"].item(), 1)
        self.assertEqual(group.peak_force.item(), 100)
        self.assertEqual(group.peak_moment.item(), 10)
        group.hold.zero_()
        position[:, 2] = 0
        _, done, _ = PlantGroup.step_outcome(group, actions)
        self.assertFalse(done.item())
        group.length.fill_(math.ceil(cfg.episode_seconds / 0.02))
        _, done, info = PlantGroup.step_outcome(group, actions)
        self.assertTrue(done.item())
        self.assertEqual(info["reason"].item(), 4)
        group.hold.fill_(4)
        position[:, 0] = cfg.workspace_xy_radius_m + 0.001
        _, _, info = PlantGroup.step_outcome(group, actions)
        self.assertEqual(info["reason"].item(), 3)
        group.cfg = cfg
        group.peak_force.zero_()
        group.peak_moment.zero_()
        group.length.zero_()
        position[:, 0] = 0
        position[:, 2] = -0.0147
        group.hold.zero_()
        group.minimum_peg_height_offset = lambda q: torch.full((q.shape[0],), -0.001)
        _, done, _ = PlantGroup.step_outcome(group, actions)
        self.assertFalse(done.item())
        self.assertEqual(group.hold.item(), 0)
        self.assertEqual(group.last_reward_terms["completion_bonus"].item(), 0)
        expected = self.terms(z=-0.0157, tilt=math.degrees(0.01), yaw=math.degrees(0.02))
        for key in ("position_penalty", "orientation_penalty"):
            self.assertAlmostEqual(float(group.last_reward_terms[key]), float(expected[key]))

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
                       {"wrench_bias_distribution": "invalid"},
                       {"initial_z_min": 0.}, {"initial_z_min": 0.02},
                       {"initial_z_max": float("nan")}, {"initial_z_max": 0.03}):
            with self.assertRaises(ValueError):
                InsertionCfg(**kwargs)
        old = InsertionCfg.from_checkpoint({"envs_per_shape": 512})
        self.assertEqual(old.force_bias, 0.)
        self.assertEqual(old.wrench_bias_distribution, "per_axis")
        self.assertEqual(old.wrench_hysteresis_fraction, 0.)
        self.assertEqual(InsertionCfg().wrench_hysteresis_fraction, 0.001)
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

    def test_wrench_bias_vector_amplitude_and_legacy_sampling(self):
        group = SimpleNamespace(
            cfg=InsertionCfg(), device="cpu", generator=torch.Generator().manual_seed(42))
        group.uniform = lambda *args: PlantGroup.uniform(group, *args)
        for amplitude in (0.5, 0.1):
            samples = PlantGroup.sample_wrench_bias(group, 100000, amplitude)
            radii = samples.norm(dim=-1)
            self.assertLessEqual(float(radii.max()), amplitude+1e-7)
            self.assertAlmostEqual(float(radii.mean()), amplitude/2, delta=amplitude*0.005)
            torch.testing.assert_close(
                torch.quantile(radii, torch.tensor([0.1, 0.5, 0.9])),
                amplitude*torch.tensor([0.1, 0.5, 0.9]), atol=amplitude*0.005, rtol=0)
            directions = samples/radii[:, None]
            torch.testing.assert_close(
                directions.mean(0), torch.zeros(3), atol=0.01, rtol=0)
            torch.testing.assert_close(
                directions.T@directions/len(samples), torch.eye(3)/3, atol=0.01, rtol=0)
        torch.testing.assert_close(
            PlantGroup.sample_wrench_bias(group, 10, 0.), torch.zeros(10, 3))
        group.cfg = replace(group.cfg, wrench_bias_distribution="per_axis")
        state = group.generator.get_state()
        samples = PlantGroup.sample_wrench_bias(group, 100, 0.5)
        group.generator.set_state(state)
        torch.testing.assert_close(samples, group.uniform(100, 3, -0.5, 0.5), rtol=0, atol=0)

    def test_play_hysteresis_tracks_direction_with_bounded_loop(self):
        state = torch.zeros(1, 1, dtype=torch.float64)
        half_width = torch.tensor([[0.015]], dtype=torch.float64)
        outputs = []
        for signal in (0., 0.03, 0.06, 0.09, 0.06, 0.03, 0.):
            value = torch.tensor([[signal]], dtype=torch.float64)
            state = play_hysteresis_step(state, value, half_width)
            outputs.append(float(state))
            self.assertLessEqual(float((state-value).abs()), float(half_width))
        self.assertAlmostEqual(outputs[2], 0.045)
        self.assertAlmostEqual(outputs[4], 0.075)
        self.assertAlmostEqual(outputs[4]-outputs[2], 2*float(half_width))
        signal = torch.tensor([[-0.1, 0., 0.1]], dtype=torch.float64)
        torch.testing.assert_close(
            play_hysteresis_step(torch.ones_like(signal), signal, torch.zeros_like(signal)),
            signal, rtol=0, atol=0)

    def test_zero_preinsertion_tilt_cost_and_depth_ramp(self):
        self.assertEqual(InsertionCfg().preinsertion_tilt_weight, 0.)
        for z in (0.004, 0.):
            self.assertEqual(float(self.terms(z=z, tilt=10.)["orientation_penalty"]), 0.)
        for depth in (0.005, 0.015, 0.0225):
            for inserted_mm in (0., 1., 2., 3., 4., 6.):
                with self.subTest(depth=depth, inserted_mm=inserted_mm):
                    penalty = self.terms(
                        z=-inserted_mm*0.001, depth=depth, tilt=5.)["orientation_penalty"]
                    weight = min(inserted_mm/4., 1.)
                    self.assertAlmostEqual(float(penalty), -0.01*weight)
            expected = -0.02*2.*2.*math.tanh(((depth-0.002)/depth)/2.)
            self.assertAlmostEqual(
                float(self.terms(z=-0.002, depth=depth)["position_penalty"]), expected)
        for z in (0.004, 0., -0.0075, -0.015):
            self.assertAlmostEqual(
                float(self.terms(z=z, yaw=5.)["orientation_penalty"]), -0.01)
        saved = InsertionCfg.from_checkpoint({"preinsertion_tilt_weight": 0.25})
        self.assertEqual(saved.preinsertion_tilt_weight, 0.25)

    def test_minimum_peg_height_matches_geometry(self):
        q = rotvec_quat(torch.tensor([
            [0., 0., 0.], [0.2, -0.3, 0.4], [math.pi, 0., 0.]],
            dtype=torch.float64))
        for shape in (*TRAIN_SHAPES, "circle"):
            with self.subTest(shape=shape):
                g = geometry(shape)
                group = SimpleNamespace(
                    shape=shape, geometry=g, clearance=g.clearance,
                    peg_polygon=torch.tensor(g.peg_polygon, dtype=q.dtype))
                actual = PlantGroup.minimum_peg_height_offset(group, q)
                if shape == "circle":
                    angles = torch.arange(8192, dtype=q.dtype)*2*math.pi/8192
                    polygon = (g.radius-g.clearance/2)*torch.stack(
                        (angles.cos(), angles.sin()), dim=-1)
                else:
                    polygon = group.peg_polygon
                bottom = torch.cat((polygon, torch.zeros(len(polygon), 1, dtype=q.dtype)), dim=-1)
                vertices = torch.cat((bottom, bottom + bottom.new_tensor([0., 0., g.peg_length])))
                expected = tip_pose(
                    torch.zeros_like(vertices), q[:, None, :], vertices[None, :, :])[:, :, 2].amin(-1)
                torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)
                self.assertAlmostEqual(float(actual[0]), 0.)
                self.assertLess(float(actual[1]), 0.)

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
        # 孔口附近保留居中收益，但平方 XY 比原线性 XY 更宽松
        mouth_delta = (float(self.terms(z=0., xy=0.)["position_penalty"])
                       - float(self.terms(z=0., xy=0.0002)["position_penalty"]))
        self.assertGreater(mouth_delta, 0.)
        linear_delta = 0.08*(math.tanh(1.08/2)-math.tanh(1./2))
        self.assertLess(mouth_delta, linear_delta)
        self.assertEqual(self.reward(z=0.001, tilt=5.), hover)
        self.assertLess(self.reward(z=0.001, yaw=5.), self.reward(z=0.001, tilt=5.))
        self.assertLess(float(self.terms(z=-0.013, tilt=3.)["orientation_penalty"]),
                        float(self.terms(z=-0.002, tilt=3.)["orientation_penalty"]))
        self.assertEqual(float(self.terms(force=5.)["force_penalty"]), 0)
        self.assertLess(float(self.terms(force=6.)["force_penalty"]), 0)
        worst = self.reward(z=1000., xy=1000., tilt=180., yaw=180.,
                            force=1000., moment=1000., action=100., previous=-100.)
        self.assertGreaterEqual(worst, -cfg.completion_step_cost_bound)
        self.assertLess(self.reward(failure=True), -cfg.completion_step_cost_bound / (1 - PPO_DISCOUNT))

    def test_squared_xy_and_linear_remaining_depth(self):
        for xy in (0., 0.0002, 0.001, 0.0025, 0.005):
            for z in (0.004, 0., -0.0075, -0.015, -0.02):
                with self.subTest(xy=xy, z=z):
                    metric = (xy/0.0025)**2 + max(z+0.015, 0.)/0.015
                    expected = -0.08*math.tanh(metric/2.)
                    self.assertAlmostEqual(
                        float(self.terms(xy=xy, z=z)["position_penalty"]), expected)

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
        self.assertTrue(bool((before[:, :3].norm(dim=-1) <= 0.25).all()))
        self.assertTrue(bool((before[:, 3:].norm(dim=-1) <= 0.005).all()))
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
        self.assertEqual(group.sim.wp_data.naconmax, group.n * 64)
        self.assertEqual(group.sim.wp_data.njmax, 256)
        self.assertAlmostEqual(group.difficulty, 0.25)
        env.step(torch.zeros(4, 6, device=self.device))
        self.assertIsNotNone(group.physics_rollout_graph)
        self.assertIsNotNone(group.reset_constants_graph)
        self.assertIsNotNone(group.action_graph)
        self.assertIsNotNone(group.outcome_graph)
        env.noise_curriculum.windows["square"].extend([True] * 996)
        env.record_curriculum_outcomes(torch.tensor([[0, 1, 1, 1]] * 4, device=self.device))
        logs = env.advance_noise_curriculum()
        self.assertEqual(group.clearance, 0.001)
        self.assertEqual(group.sim.wp_data.naconmax, group.n * 64)
        self.assertEqual(group.sim.wp_data.njmax, 256)
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
