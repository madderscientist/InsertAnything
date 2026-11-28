"""奖励函数、孔间隙课程与批量 GPU 物理环境测试"""

import math
from dataclasses import asdict, replace
from itertools import product
from typing import Any
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import torch

from insertion import (InsertionCfg, InsertionEnv, PPO_DISCOUNT,
                       ADRCurriculum, SuccessNoiseCurriculum, TRAIN_SHAPES, bounded_stiffness_step,
                       completion_cost_terms, geometry, quat_rotvec, tip_pose)
from insertion.config import REAL2SIM_VALUES
from insertion.curriculum import adaptive_environment_counts
from insertion.transforms import quat_inv, rotate, rotvec_quat
from insertion.env import PlantGroup, action_costs, lowpass_velocity, play_hysteresis_step
from insertion.shapes.base import PreparedGeometry
from real2sim.evaluate_real2sim import sign_flips, velocity_reversals
from real2sim.train_real2sim import real2sim_config


class RewardAndCurriculumUnitTests(unittest.TestCase):
    def test_curriculum_waits_for_full_cohort_including_slow_failures(self):
        curriculum = SuccessNoiseCurriculum(("square",))
        env = SimpleNamespace(
            noise_curriculum=curriculum, curriculum_pending=[], groups=[],
            curriculum_cohort=None, curriculum_cohort_seen=None, curriculum_cohort_armed=None)
        batch = torch.zeros(1000, 5)
        batch[:, 3] = 1
        # 起始回合不计入，先让全部环境结束一次
        batch[:, 1] = 1
        env.curriculum_pending.append(batch.clone())
        batch[:, 1] = 0
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
        # 跨越队列起点的回合不计入，下一批才是队列开始后的完整回合
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(curriculum.level, 0)
        env.curriculum_pending.append(batch.clone())
        InsertionEnv.advance_noise_curriculum(env)
        self.assertEqual(curriculum.level, 1)
        self.assertEqual(len(curriculum.windows["square"]), 0)

    def test_adr_cohort_logs_boundary_success_per_dimension(self):
        env = SimpleNamespace(
            noise_curriculum=ADRCurriculum(("square",), 0.002, 0.0001), curriculum_pending=[], groups=[],
            curriculum_cohort=None, curriculum_cohort_seen=None, curriculum_cohort_armed=None)
        batch = torch.zeros(1000, 5)
        batch[:, 1] = batch[:, 3] = 1
        batch[:100, 4], batch[100:200, 4] = 1, 2
        env.curriculum_pending.append(batch.clone())
        self.assertNotIn("curriculum/boundary_success/noise/square", InsertionEnv.advance_noise_curriculum(env))
        batch[:70, 2] = batch[100:150, 2] = 1
        env.curriculum_pending.append(batch)
        logs = InsertionEnv.advance_noise_curriculum(env)
        self.assertAlmostEqual(float(logs["curriculum/boundary_success/noise/square"]), 0.7)
        self.assertAlmostEqual(float(logs["curriculum/boundary_success/clearance/square"]), 0.5)
        # 成功率窗口含边界回合
        self.assertAlmostEqual(float(logs["curriculum/success/square"]), 0.12)

    def test_curriculum_cohort_excludes_episodes_in_progress_at_start(self):
        curriculum = SuccessNoiseCurriculum(("square",))
        env = SimpleNamespace(
            noise_curriculum=curriculum, curriculum_pending=[], groups=[],
            curriculum_cohort=None, curriculum_cohort_seen=None, curriculum_cohort_armed=None)

        def step(done, success):
            batch = torch.zeros(1000, 5)
            batch[:, 3] = 1
            batch[done, 1] = 1
            batch[done & success, 2] = 1
            env.curriculum_pending.append(batch)
            InsertionEnv.advance_noise_curriculum(env)

        everyone = torch.ones(1000, dtype=torch.bool)
        nobody = torch.zeros(1000, dtype=torch.bool)
        fast = torch.arange(1000) < 700
        # 起始回合计时被错开，不计入
        step(fast, nobody)
        step(~fast, everyone)
        self.assertEqual(len(curriculum.windows["square"]), 0)
        self.assertFalse(env.curriculum_cohort_seen.any())
        step(fast, everyone)
        self.assertEqual(curriculum.level, 0)
        step(~fast, everyone)
        self.assertEqual(curriculum.level, 1)

    def test_depth_success_hold_reset_and_safety(self):
        cfg = InsertionCfg(success_opening_required=False)
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
        group.cfg = replace(cfg, terminate_on_overload=True)
        for field, value, reason in (("peak_force", 31., 2), ("peak_moment", 2.1, 2)):
            group.hold.fill_(4)
            getattr(group, field).fill_(value)
            _, done, info = PlantGroup.step_outcome(group, actions)
            self.assertTrue(done.item())
            self.assertEqual(info["reason"].item(), reason)
            self.assertEqual(group.last_reward_terms["completion_bonus"].item(), 0)
            getattr(group, field).zero_()
        # 默认训练模式超载不终止，仅重置成功保持计数
        group.cfg = cfg
        for field, value in (("peak_force", 31.), ("peak_moment", 2.1)):
            group.hold.fill_(4)
            getattr(group, field).fill_(value)
            _, done, info = PlantGroup.step_outcome(group, actions)
            self.assertFalse(done.item())
            self.assertEqual(group.hold.item(), 0)
            self.assertEqual(info["reason"].item(), 0)
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
                       {"stiction_force": -0.1}, {"stiction_moment": float("nan")},
                       {"wall_friction_force": -1.}, {"action_cost_cap": 0.},
                       {"initial_z_min": 0.}, {"initial_z_min": 0.06},
                       {"initial_z_max": float("nan")}, {"initial_z_max": 0.07},
                       {"mass_min_kg": 2.0}, {"stiction_force_z": -1.0},
                       {"target_lowpass_translation_rad_s": 0.0}):
            with self.assertRaises(ValueError):
                InsertionCfg(**kwargs)
        old = InsertionCfg.from_checkpoint({"envs_per_shape": 512})
        self.assertEqual(old.force_bias, 0.)
        self.assertEqual((old.stiction_force, old.stiction_moment), (0., 0.))
        self.assertEqual(old.wall_friction_force, 0.)
        self.assertEqual(old.action_cost_cap, 4.)
        self.assertEqual((InsertionCfg().stiction_force, InsertionCfg().stiction_moment), (0.6, 0.06))
        self.assertEqual((InsertionCfg().wall_friction_force, InsertionCfg().action_cost_cap), (4.0, 12.0))
        self.assertEqual(old.wrench_bias_distribution, "per_axis")
        self.assertEqual(old.wrench_hysteresis_fraction, 0.)
        self.assertEqual(InsertionCfg().wrench_hysteresis_fraction, 0.001)
        self.assertEqual(old.environment_counts, (512,) * 5)
        self.assertEqual((old.initial_z_min, old.initial_z_max), (0.004, 0.02))
        self.assertEqual((old.workspace_z_max_m, old.action_change_cost_per_second), (0.03, 0.05))
        self.assertEqual((old.target_lowpass, old.stiction_force_z, old.stiction_moment_xy), (False, None, None))
        self.assertEqual((old.mass_min_kg, old.damping_ratio_min, old.down_mass_max_kg), (0.7, 0.8, 0.4))
        self.assertEqual((old.rotation_pivot_max_m, old.stick_slip_max_pp_m,
                          old.step_bias_rate_per_second), (0., 0., 0.))
        migrated = real2sim_config(asdict(old))
        self.assertEqual(migrated.envs_per_shape, 512)
        for key, expected in REAL2SIM_VALUES.items():
            self.assertEqual(getattr(migrated, key), expected)

    def terms(self, z=0.001, xy=0., depth=0.015, tilt=0., yaw=0., force=0.,
              moment=0., success=False, failure=False, timeout=False, action=0., previous=0.,
              cfg: InsertionCfg | None = None):
        cfg = InsertionCfg() if cfg is None else cfg
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
        for z in (0.004, 0.):
            self.assertEqual(float(self.terms(z=z, tilt=5.)["orientation_penalty"]), 0.0)
        for depth in (0.005, 0.015, 0.0225):
            for inserted_mm in (0., 1., 2., 3., 4., 6.):
                with self.subTest(depth=depth, inserted_mm=inserted_mm):
                    penalty = self.terms(
                        z=-inserted_mm*0.001, depth=depth, tilt=5.)["orientation_penalty"]
                    weight = min(inserted_mm/4., 1.)
                    self.assertAlmostEqual(float(penalty), -0.01*weight)
            expected = -0.02*2.*(1.-InsertionCfg().z_outer_cost_weight)*math.tanh((depth-0.002)/depth)
            self.assertAlmostEqual(
                float(self.terms(z=-0.002, depth=depth)["position_penalty"]), expected)
        for z in (0.004, 0., -0.0075, -0.015):
            self.assertAlmostEqual(
                float(self.terms(z=z, yaw=5.)["orientation_penalty"]), -0.01)
        for rough_surface in (False, True):
            with self.subTest(legacy_rough_surface=rough_surface):
                cfg = InsertionCfg.from_checkpoint({
                    "rough_surface": rough_surface,
                    "surface_z_free_m": 0.006 if rough_surface else 0.0,
                    "preinsertion_tilt_weight": 0.5,
                })
                self.assertEqual(float(self.terms(z=0.004, tilt=5., cfg=cfg)["orientation_penalty"]), 0.0)
                self.assertAlmostEqual(
                    float(self.terms(z=0.004, tilt=5., yaw=5., cfg=cfg)["orientation_penalty"]), -0.01)

    def test_surface_z_free_band_exempts_obstacle_lift_only_in_band(self):
        cfg_free = InsertionCfg(surface_z_free_m=0.005)
        cfg_base = InsertionCfg(surface_z_free_m=0.0)
        depth = torch.tensor([0.015], dtype=torch.float64)
        zero1 = torch.zeros(1, dtype=torch.float64)
        zero6 = torch.zeros(1, 6, dtype=torch.float64)
        flag = torch.tensor([False])

        def pos_pen(cfg, z_val):
            return float(completion_cost_terms(
                cfg, zero1, torch.tensor([z_val + 0.015], dtype=torch.float64),
                depth, 0.01, zero1, zero1, zero6, zero6, zero6, flag, flag, flag,
            )["position_penalty"])

        at_0 = pos_pen(cfg_free, 0.0)
        self.assertAlmostEqual(at_0, pos_pen(cfg_base, 0.0))
        for z_band in (0.001, 0.0025, 0.005):
            self.assertAlmostEqual(pos_pen(cfg_free, z_band), at_0)
            self.assertLess(pos_pen(cfg_base, z_band), at_0)
        self.assertLess(pos_pen(cfg_free, 0.010), at_0)
        self.assertAlmostEqual(pos_pen(cfg_free, -0.006), pos_pen(cfg_base, -0.006))

    def test_retired_lift_checkpoint_fields_are_stripped(self):
        cfg = InsertionCfg.from_checkpoint({
            "stagnation_cost_per_second": 2.5,
            "stagnation_lift_cost_per_second": 0.0,
            "stagnation_lift_target": 5.5,
            "stagnation_lift_scale": 48.0,
            "stagnation_lateral_target": 4.0,
            "stagnation_lateral_scale": 24.0,
            "safety_failure_penalty": 400.0,
        })
        self.assertEqual(cfg.stagnation_cost_per_second, 2.5)
        self.assertFalse(hasattr(cfg, "stagnation_lift_cost_per_second"))

    def test_minimum_peg_height_matches_geometry(self):
        q = rotvec_quat(torch.tensor([
            [0., 0., 0.], [0.2, -0.3, 0.4], [math.pi, 0., 0.]],
            dtype=torch.float64))
        for shape in (*TRAIN_SHAPES, "circle"):
            with self.subTest(shape=shape):
                g = geometry(shape)
                peg_polygon = torch.tensor(g.peg_polygon, dtype=q.dtype)
                group = SimpleNamespace(
                    shape=shape, geometry=g, clearance=g.clearance, peg_polygon=peg_polygon,
                    geo=PreparedGeometry(g, torch.tensor(g.hole_polygon, dtype=q.dtype), peg_polygon),
                    shape_params=q.new_tensor(g.definition.nominal_params(g)).expand(len(q), -1))
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
        self.assertFalse(hasattr(InsertionCfg.from_checkpoint({"load_cost_power": 1}), "load_cost_power"))

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
        self.assertLess(float(self.terms(action=3.)["action_penalty"]),
                        float(self.terms(action=1.)["action_penalty"]))
        self.assertLess(float(self.terms(action=3., previous=-1.)["action_change_penalty"]),
                        float(self.terms(action=1., previous=-1.)["action_change_penalty"]))
        worst = self.reward(z=1000., xy=1000., tilt=180., yaw=180.,
                            force=1000., moment=1000., action=100., previous=-100.)
        self.assertGreaterEqual(worst, -cfg.completion_step_cost_bound)
        self.assertLess(self.reward(failure=True), -cfg.hold_cost_bound)

    def test_free_air_action_change_squared(self):
        cfg = replace(InsertionCfg(), free_air_action_change_squared=True)
        actions = torch.tensor([[0.5, 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0]] * 2)
        contact = torch.tensor([False, False, True, True])
        _, cost = action_costs(cfg, actions, torch.zeros_like(actions), contact)
        change = cfg.action_cost_cap * torch.tanh(actions[:, 0] / cfg.action_cost_cap)
        torch.testing.assert_close(cost, torch.cat((
            cfg.free_air_action_change_cost_per_second * change[:2].square(),
            cfg.action_change_cost_per_second * change[2:])))
        self.assertGreater(cfg.max_action_change_cost_per_second,
                           InsertionCfg().max_action_change_cost_per_second)

    def test_squared_xy_and_linear_remaining_depth(self):
        cfg, legacy_cfg = InsertionCfg(), InsertionCfg.from_checkpoint({})
        self.assertEqual(legacy_cfg.xy_outer_cost_weight, 0.0)
        self.assertEqual(legacy_cfg.z_outer_cost_weight, 0.0)
        w_z, ceiling = cfg.z_outer_cost_weight, cfg.workspace_z_max_m
        eps, r_max = 0.05 * 0.01, cfg.workspace_xy_radius_m
        norm = r_max * r_max / (math.hypot(r_max, eps) + eps)
        radii = (0., 0.0002, 0.001, 0.0025, 0.005, 0.0075, 0.010, 0.020, 0.035)
        for xy in radii:
            x = min(xy, r_max)
            outer = x * x / ((math.hypot(x, eps) + eps) * norm)
            fine = math.tanh((xy / 0.0025) ** 2)
            xy_term = (1.0 - cfg.xy_outer_cost_weight) * fine + cfg.xy_outer_cost_weight * outer
            for z in (0.004, 0., -0.0075, -0.015, -0.02):
                with self.subTest(xy=xy, z=z):
                    z_term = math.tanh(max(z + 0.015, 0.) / 0.015)
                    z_outer = min(max(z, 0.), ceiling) / ceiling
                    self.assertAlmostEqual(
                        float(self.terms(xy=xy, z=z, cfg=cfg)["position_penalty"]),
                        -0.04 * (xy_term + (1.0 - w_z) * z_term + w_z * z_outer))
                    self.assertAlmostEqual(
                        float(self.terms(xy=xy, z=z, cfg=legacy_cfg)["position_penalty"]),
                        -0.04 * (fine + z_term))
            self.assertGreater(
                float(self.terms(xy=xy, z=0.001, cfg=cfg)["position_penalty"]),
                float(self.terms(xy=xy, z=0.020, cfg=cfg)["position_penalty"]))
            self.assertGreater(
                float(self.terms(xy=xy, z=0.040, cfg=cfg)["position_penalty"]),
                float(self.terms(xy=xy, z=0.065, cfg=cfg)["position_penalty"]))
        xy_f32 = torch.tensor(radii + (0.050,), dtype=torch.float32)
        z1 = torch.zeros_like(xy_f32)
        z6 = torch.zeros(len(xy_f32), 6, dtype=torch.float32)
        zb = torch.zeros(len(xy_f32), dtype=torch.bool)
        penalties_f32 = completion_cost_terms(
            cfg, xy_f32, z1, torch.full_like(xy_f32, 0.015), 0.01, z1, z1, z6, z6, z6, zb, zb, zb,
        )["position_penalty"]
        self.assertTrue(bool((penalties_f32[1:-1] < penalties_f32[:-2]).all()))
        self.assertAlmostEqual(float(penalties_f32[-1]), float(penalties_f32[-2]), places=6)
        for bad_weight in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                InsertionCfg(xy_outer_cost_weight=bad_weight)
            with self.assertRaises(ValueError):
                InsertionCfg(z_outer_cost_weight=bad_weight)

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
        group = SimpleNamespace(
            gain_q=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
            k=torch.tensor([[200., 200., 200., 4., 4., 4.]], dtype=torch.float64),
            d=torch.tensor([[10., 10., 10., 1., 1., 1.]], dtype=torch.float64),
            stiction=torch.tensor([[0.6, 0.6, 0.6, 0.06, 0.06, 0.06]], dtype=torch.float64))
        zero_v = torch.zeros(1, 3, dtype=torch.float64)
        inside = PlantGroup.impedance(group, torch.tensor([[0.002, -0.003, 0.]], dtype=torch.float64), zero_v, slice(0, 3))
        torch.testing.assert_close(inside, zero_v)
        outside = PlantGroup.impedance(
            group, torch.tensor([[0.005, -0.005, 0.]], dtype=torch.float64),
            torch.tensor([[0.01, 0., 0.]], dtype=torch.float64), slice(0, 3))
        torch.testing.assert_close(outside, torch.tensor([[0.3, -0.4, 0.]], dtype=torch.float64))

    def test_target_lowpass_matches_critically_damped_step(self):
        for omega in (30.0, 50.0):
            x, v = torch.zeros(1), torch.zeros(1)
            for step in range(1, 61):
                v = lowpass_velocity(v, 1.0 - x, omega, 0.001)
                x = x + v * 0.001
                if step in (40, 60):
                    t = step * 0.001
                    self.assertAlmostEqual(float(x), 1 - (1 + omega*t)*math.exp(-omega*t), delta=0.02)

    def test_action_flip_and_velocity_reversal_metrics(self):
        flips, pairs = sign_flips(torch.tensor([[1., -1., 0.05]]), torch.tensor([[-1., -2., -1.]]))
        self.assertEqual((int(flips), int(pairs)), (1, 2))
        reversed_, valid = velocity_reversals(torch.tensor([[1e-3, 0.], [1e-3, 0.], [1e-5, 0.]]),
                                              torch.tensor([[-1e-3, 0.], [0., 1e-3], [-1e-3, 0.]]))
        self.assertEqual(reversed_.tolist(), [True, False, False])
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_adr_adjusts_noise_and_clearance_independently(self):
        curriculum = ADRCurriculum(("square", "triangle"), 0.002, 0.0001)
        full = {"square": [True] * 10, "triangle": [True] * 10}
        none: tuple[list[bool], list[bool]] = ([], [])
        self.assertEqual((curriculum.clearance_for("square"), curriculum.difficulty), (0.002, 0.))
        self.assertFalse(curriculum.update(full, {"square": ([True] * 199, []), "triangle": none}))
        self.assertTrue(curriculum.update(full, {"square": ([True], []), "triangle": none}))
        self.assertEqual(curriculum.phi, {"square": [0.05, 0.], "triangle": [0., 0.]})
        self.assertEqual(curriculum.difficulty, 0.)
        # 间隙维单独计数；缓冲带内不调节，已在下限时不再回退
        self.assertFalse(curriculum.update(full, {"square": ([], [True] * 130 + [False] * 70), "triangle": none}))
        self.assertFalse(curriculum.update(full, {"square": ([], [False] * 200), "triangle": none}))
        for _ in range(20):
            curriculum.update(full, {"square": ([True] * 200, [True] * 200),
                                     "triangle": ([True] * 200, [True] * 200)})
        self.assertEqual(curriculum.phi, {"square": [1.05, 1.], "triangle": [1., 1.]})
        self.assertEqual((curriculum.difficulty, curriculum.scale_for("square")), (1., 1.))
        self.assertAlmostEqual(curriculum.noise_floor_for("square"), 0.05)
        self.assertAlmostEqual(curriculum.clearance_for("triangle"), 0.0001)
        self.assertEqual((curriculum.noise_floor_for("triangle"), curriculum.clearance_ceiling_for("triangle")),
                         (0., 0.002))
        # 难侧满后易侧边界向难侧收拢，直到整维钉在最难值
        for _ in range(20):
            curriculum.update(full, {"square": ([True] * 200, [True] * 200),
                                     "triangle": ([True] * 200, [True] * 200)})
        self.assertEqual(curriculum.phi, {"square": [2., 2.], "triangle": [2., 2.]})
        self.assertEqual((curriculum.difficulty, curriculum.scale_for("square"),
                          curriculum.noise_floor_for("square")), (1., 1., 1.))
        self.assertAlmostEqual(curriculum.clearance_for("square"), 0.0001)
        self.assertAlmostEqual(curriculum.clearance_ceiling_for("square"), 0.0001)
        self.assertTrue(curriculum.update(full, {"square": ([], [False] * 200), "triangle": none}))
        self.assertEqual(curriculum.phi["square"], [2., 1.95])
        self.assertAlmostEqual(curriculum.clearance_for("square"), 0.0001)
        self.assertAlmostEqual(curriculum.clearance_ceiling_for("square"), 0.002 * 0.05 ** 0.95)
        restored = ADRCurriculum(("square", "triangle"), 0.002, 0.0001)
        restored.load_state_dict(curriculum.state_dict())
        self.assertEqual(restored.state_dict(), curriculum.state_dict())
        cfg = InsertionCfg(noise_curriculum=True, noise_curriculum_mode="adr",
                           shape_randomization=True, shapes=("triangle",))
        self.assertEqual(cfg.noise_scale(0), 0.)
        with self.assertRaises(ValueError):
            InsertionCfg(noise_curriculum_mode="adr")
        with self.assertRaises(ValueError):
            InsertionCfg(noise_curriculum=True, noise_curriculum_mode="adr",
                         shape_randomization=True, shapes=("square",))

    def test_clearance_difficulty_and_noise_curriculum(self):
        levels = (0.002, 0.001, 0.0005, 0.00025, 0.00015, 0.0001)
        curriculum = SuccessNoiseCurriculum(("square", "triangle"), clearances=levels, noise_levels=10)
        self.assertEqual(curriculum.clearance, 0.002)
        self.assertAlmostEqual(curriculum.difficulty, 0.)
        self.assertEqual(curriculum.scale, 0.)
        self.assertFalse(curriculum.update({"square": [True] * 999, "triangle": [True] * 1000}))
        for i, expected_clearance in enumerate(levels[1:], start=1):
            self.assertTrue(curriculum.update({"square": [True] * 1000, "triangle": [True] * 1000}))
            self.assertEqual(curriculum.clearance, expected_clearance)
            self.assertAlmostEqual(curriculum.difficulty, i / 5)
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
            rotation_bias=0,
            position_noise=0, rotation_noise=0, force_noise=0, moment_noise=0,
            initial_z_max=0.02, target_lowpass=False,
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
            z_max = 0.001 + 0.009 * difficulty
            self.assertTrue(bool(((tip[:, 2] >= 0.001-1e-7) & (tip[:, 2] <= z_max+1e-7)).all()))
            self.assertLess(float(tip[:, 2].min()), 0.001 + 0.1 * (z_max - 0.001))
            self.assertGreater(float(tip[:, 2].max()), z_max - 0.1 * (z_max - 0.001))
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
        self.assertAlmostEqual(group.difficulty, 0.)
        env.step(torch.zeros(4, 6, device=self.device))
        self.assertIsNotNone(group.physics_rollout_graph)
        self.assertIsNotNone(group.reset_constants_graph)
        self.assertIsNotNone(group.action_graph)
        self.assertIsNotNone(group.outcome_graph)
        # 起始回合不计入，先让全部环境结束一次
        env.record_curriculum_outcomes(torch.tensor([[0, 1, 0, 1, 0]] * 4, device=self.device))
        env.advance_noise_curriculum()
        env.noise_curriculum.windows["square"].extend([True] * 996)
        env.record_curriculum_outcomes(torch.tensor([[0, 1, 1, 1, 0]] * 4, device=self.device))
        logs = env.advance_noise_curriculum()
        self.assertEqual(group.clearance, 0.001)
        self.assertEqual(group.sim.wp_data.naconmax, group.n * 64)
        self.assertEqual(group.sim.wp_data.njmax, 256)
        self.assertIsNone(group.physics_rollout_graph)
        self.assertIsNone(group.reset_constants_graph)
        self.assertIsNone(group.action_graph)
        self.assertIsNone(group.outcome_graph)
        self.assertAlmostEqual(group.difficulty, 0.2)
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

    def test_isolated_divergence_terminates_only_that_environment(self):
        cfg = replace(self.cfg, shapes=("square",), envs_per_shape=128,
                      physics_cuda_graph=True, observation_cuda_graph=True)
        env = InsertionEnv(cfg, self.device)
        actions = torch.zeros(env.num_envs, 6, device=self.device)
        env.step(actions)
        env.groups[0].sim.data.qpos[3, 0] = float("nan")
        obs, reward, done, info = env.step(actions)
        self.assertTrue(bool(torch.isfinite(obs["actor"]).all() & torch.isfinite(reward).all()))
        self.assertEqual(done.nonzero().flatten().tolist(), [3])
        self.assertEqual(int(info["terminal"]["reason"][3]), 5)
        self.assertEqual(float(reward[3]), -cfg.safety_failure_penalty)
        self.assertEqual(float(info["log"]["episode/simulation_failures"]), 1.0)
        self.assertTrue(bool(torch.isfinite(env.groups[0].sim.data.qpos[:]).all()))

    def test_rough_surface_no_timeout_and_cuda_graph_equivalence(self):
        cfg = replace(
            self.cfg,
            shapes=("square", "l"),
            envs_per_shape=4,
            episode_seconds=0.1,
            enable_timeout=False,
            timeout_penalty=0.0,
            rough_surface=True,
            rough_low_spawn_prob=0.5,
            rough_max_amplitude_m=0.004,
            workspace_z_max_m=0.06,
            surface_z_free_m=0.006,
            rough_flat_probability=0.0,
            grasp_position_error=0.0015,
            grasp_z_error=0.005,
            hole_depth_m=0.028,
            target_depth_min_m=0.003,
            target_depth_max_fraction=25.0 / 28.0,
            align_goal_z_to_observation=True,
            rotation_bias=math.radians(1.5),
            stagnation_cost_per_second=2.5,
            safety_failure_penalty=250.0,
            dynamics="train",
            training=False,
        )
        eager = InsertionEnv(cfg, self.device)
        captured = InsertionEnv(
            replace(cfg, physics_cuda_graph=True, observation_cuda_graph=True),
            self.device,
        )
        for g in eager.groups:
            self.assertTrue(bool((g.surface_max_height > 0).all()))
            self.assertGreater(float(g.surface_max_height.std()), 0.0)
        zero_actions = torch.zeros(eager.num_envs, 6, device=self.device)
        with torch.inference_mode():
            for _ in range(8):
                obs_e, rew_e, done_e, info_e = eager.step(zero_actions)
                obs_c, rew_c, done_c, info_c = captured.step(zero_actions)
                self.assertFalse(bool(done_e.any()))
                self.assertFalse(bool(info_e["terminal"]["time_outs"].any()))
                torch.testing.assert_close(obs_c, obs_e, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(rew_c, rew_e, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(done_c, done_e)
            self.assertTrue(bool((eager.groups[0].last_reward_terms["stagnation_penalty"] < 0).all()))
            self.assertTrue(bool((eager.episode_length_buf == 8).all()))
            down_actions = torch.zeros_like(zero_actions)
            down_actions[:, 2] = -2.0
            for step_i in range(10):
                down_actions[:, 0] = 0.5 * math.sin(step_i)
                obs_e, rew_e, done_e, _ = eager.step(down_actions)
                obs_c, rew_c, done_c, _ = captured.step(down_actions)
                torch.testing.assert_close(obs_c, obs_e, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(rew_c, rew_e, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(done_c, done_e)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Real2SimPlantTests(unittest.TestCase):
    device = "cuda:0"

    def test_real_plant_matches_measured_step_and_z_deadband(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=1, training=False, dynamics="real",
                           tip_calibration_error=0., initial_tilt=0., target_lowpass=True,
                           initial_yaw=0., initial_z_min=0.04, initial_z_max=0.05, workspace_z_max_m=0.07)
        group = InsertionEnv(cfg, self.device).groups[0]
        # 0.1 kg 表观质量的实机应紧跟 20 ms 延迟后的部署低通轨迹，偏差不超过 0.3 N 死区 (约 0.15) 加余量
        ideal = [0.] + [1 - (1 + 30*t) * math.exp(-30*t) for t in (0.02*i for i in range(1, 10))]
        for axis, step in ((0, 0.003), (2, 0.003)):
            group.reset(torch.arange(1, device=self.device))
            group.slip_amplitude.zero_()
            nominal = group.nominal_offset + group.calibration
            start = tip_pose(*group.state()[:2], nominal)
            goal = start.clone()
            goal[:, axis] += step
            fractions = []
            with torch.inference_mode():
                for _ in range(10):
                    actions = torch.zeros(1, 6, device=self.device)
                    actions[:, :3] = (goal - tip_pose(*group.state()[:2], nominal)) / cfg.translation_limit
                    group.prepare_actions(actions)
                    group.advance_physics()
                    fractions.append(float((tip_pose(*group.state()[:2], nominal) - start)[0, axis]) / step)
            if axis == 0:
                self.assertLess(fractions[0], 0.05, fractions)
                self.assertLess(max(abs(a - b) for a, b in zip(fractions, ideal)), 0.2, (fractions, ideal))
            else:
                self.assertLess(max(abs(f) for f in fractions), 0.05, fractions)

    def test_lowpass_off_episode_follows_light_mass_impedance_step(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=1, training=False, dynamics="real",
                           tip_calibration_error=0., initial_tilt=0., initial_yaw=0.,
                           initial_z_min=0.04, initial_z_max=0.05, workspace_z_max_m=0.07,
                           target_lowpass=True, target_lowpass_off_probability=1.0)
        group = InsertionEnv(cfg, self.device).groups[0]
        group.reset(torch.arange(1, device=self.device))
        group.slip_amplitude.zero_()
        torch.testing.assert_close(group.lowpass_omega, torch.full((1, 3), math.sqrt(6500.), device=self.device))
        nominal = group.nominal_offset + group.calibration
        start = tip_pose(*group.state()[:2], nominal)
        goal = start.clone()
        goal[:, 0] += 0.003
        fractions = []
        with torch.inference_mode():
            for _ in range(8):
                actions = torch.zeros(1, 6, device=self.device)
                actions[:, :3] = (goal - tip_pose(*group.state()[:2], nominal)) / cfg.translation_limit
                group.prepare_actions(actions)
                group.advance_physics()
                fractions.append(float((tip_pose(*group.state()[:2], nominal) - start)[0, 0]) / 0.003)
        # 20 ms 延迟后由 80 rad/s、ζ=0.35 的轻质量响应追阶跃，明显快于部署低通并带超调
        self.assertLess(fractions[0], 0.05, fractions)
        self.assertGreater(fractions[2], 0.7, fractions)
        self.assertGreater(max(fractions), 1.05, fractions)

    def test_motion_vibration_is_narrowband_scaled_and_free_air_change_costlier(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=256, training=False,
                           vibration_force_min_n=1.0, vibration_force_max_n=1.0)
        group = InsertionEnv(cfg, self.device).groups[0]
        samples = []
        with torch.inference_mode():
            for step in range(400):
                group.update_vibration()
                if step >= 100:
                    samples.append(group.vibration_state[:, 0].clone())
        unit = torch.stack(samples)
        self.assertAlmostEqual(float(unit.std()), 1.0, delta=0.08)
        lag1 = float((unit[1:] * unit[:-1]).mean() / unit.square().mean())
        self.assertGreater(lag1, 0.2)
        group.sim.data.qvel[:] = 0
        group.update_vibration()
        self.assertEqual(float(group.vibration.abs().max()), 0.0)
        terms = lambda contact: completion_cost_terms(
            cfg, torch.zeros(1), torch.ones(1)*0.02, torch.ones(1)*0.01, 0.01, torch.zeros(1),
            torch.zeros(1), torch.zeros(1, 6), torch.ones(1, 6), -torch.ones(1, 6),
            torch.tensor([False]), torch.tensor([False]), torch.tensor([False]),
            in_contact=torch.tensor([contact]))["action_change_penalty"]
        self.assertAlmostEqual(float(terms(False) / terms(True)),
                               cfg.free_air_action_change_cost_per_second / cfg.action_change_cost_per_second,
                               places=4)

    def test_downstream_inertial_wrench_uses_sensor_and_offset_com_geometry(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=64, training=False)
        group = InsertionEnv(cfg, self.device).groups[0]
        group.reset(torch.arange(64, device=self.device))
        distance = (group.sensor_offset - group.nominal_offset)[:, 2]
        self.assertTrue(bool(((distance >= cfg.sensor_tip_min_m - 1e-6)
                              & (distance <= cfg.sensor_tip_max_m + 1e-6)).all()))
        self.assertEqual((real2sim_config(asdict(cfg)).sensor_tip_min_m,
                          InsertionCfg.from_checkpoint({}).sensor_tip_min_m), (0.18, None))
        a = torch.tensor([2.0, -1.0, 0.5], device=self.device)
        alpha = torch.tensor([3.0, 5.0, -2.0], device=self.device)
        group.sim.data.qfrc_constraint[:] = 0
        group.wall_force_z.zero_()
        group.sim.data.qvel[:] = 0
        group.sim.data.qacc[:, :3] = a
        group.sim.data.qacc[:, 3:6] = alpha
        with torch.inference_mode():
            group.update_sensor()
        _, q, _, _ = group.state()
        r = group.com_offset
        force = -group.down_mass * (rotate(quat_inv(q), a.expand(64, 3))
                                    + torch.linalg.cross(alpha.expand(64, 3), r))
        moment = -group.down_inertia * alpha + torch.linalg.cross(r - group.offset, force)
        torch.testing.assert_close(group.raw_wrench, torch.cat((force, moment), -1), atol=1e-4, rtol=1e-4)

    def test_hard_state_reset_restores_recorded_true_tip_pose(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=16, training=True,
                           hard_state_reset_probability=1.0)
        group = InsertionEnv(cfg, self.device).groups[0]
        group.difficulty = 1.
        stored = torch.tensor([0.004, -0.003, 0.006], device=self.device)
        group.hard_pos[:] = stored
        group.hard_q[:] = rotvec_quat(torch.tensor([[0.02, -0.01, 0.05]], device=self.device))
        group.hard_valid[:] = True
        group.reset(torch.arange(16, device=self.device))
        pos, q, _, _ = group.state()
        self.assertTrue(bool(group.hard_start.all()))
        torch.testing.assert_close(tip_pose(pos, q, group.offset), stored.expand(16, 3), atol=1e-5, rtol=0)
        evaluation = InsertionEnv(replace(cfg, training=False), self.device).groups[0]
        evaluation.hard_valid[:] = True
        evaluation.reset(torch.arange(16, device=self.device))
        self.assertFalse(bool(evaluation.hard_start.any()))

    def test_step_wrench_bias_injected_once_within_range(self):
        cfg = InsertionCfg(shapes=("square",), envs_per_shape=8, training=False,
                           force_bias=0., moment_bias=0., step_bias_rate_per_second=1e6)
        group = InsertionEnv(cfg, self.device).groups[0]
        with torch.inference_mode():
            group.step(torch.zeros(8, 6, device=self.device))
            first = group.wrench_bias.clone()
            group.step(torch.zeros(8, 6, device=self.device))
        torch.testing.assert_close(group.wrench_bias, first)
        force, moment = first[:, :3].norm(dim=-1), first[:, 3:].norm(dim=-1)
        self.assertTrue(bool(((force >= 1.0 - 1e-5) & (force <= 3.5 + 1e-5)).all()))
        self.assertTrue(bool(((moment >= 0.1 - 1e-5) & (moment <= 0.6 + 1e-5)).all()))


if __name__ == "__main__":
    unittest.main()
