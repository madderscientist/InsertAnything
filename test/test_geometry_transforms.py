"""几何建模与刚体坐标变换单元测试"""

from dataclasses import asdict, replace
import math
from typing import Any
import unittest

from mujoco._enums import mjtGeom
from mujoco._functions import mj_forward
from mujoco._structs import MjData
import numpy as np
import torch

from insertion import (ALL_SHAPES, InsertionCfg, action_target, actor_observation,
                       build_model, canonical_quat, geometry, get_shape, quat_inv,
                       quat_rotvec, rotate, rotvec_quat, transform_wrench)
from insertion.env import PlantGroup


class GeometryAndTransformTests(unittest.TestCase):
    def test_configured_mirror_axis_preserves_peg_and_hole_outlines(self):
        for name in ALL_SHAPES:
            with self.subTest(shape=name):
                g = geometry(name)
                axis = np.array(get_shape(name).mirror_axis_xy)
                reflected = 2 * np.outer(axis, axis) / (axis @ axis) - np.eye(2)
                for polygon in (g.peg_polygon, g.hole_polygon):
                    delta = (polygon @ reflected)[:, None, :] - polygon[None, :, :]
                    np.testing.assert_allclose(
                        np.linalg.norm(delta, axis=-1).min(axis=1), 0.0, atol=1e-12)

    def test_all_holes_open_and_side_contact_exists(self):
        for shape in ALL_SHAPES:
            with self.subTest(shape=shape):
                g = geometry(shape)
                model = build_model(g)
                data = MjData(model)
                data.qpos[:3] = [0, 0, 0.04]
                data.qpos[3:] = [1, 0, 0, 0]
                mj_forward(model, data)
                self.assertEqual(data.ncon, 0)
                if shape == "l":
                    displacement = np.array([g.clearance, 0])
                else:
                    edge = g.hole_polygon[1] - g.hole_polygon[0]
                    outward = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
                    displacement = outward * g.clearance
                data.qpos[:2] = displacement
                mj_forward(model, data)
                self.assertGreater(data.ncon, 0)
                self.assertLess(min(c.dist for c in data.contact), -g.clearance * 0.2)

    def test_face_clearance_notch_and_circle_error(self):
        for shape in ("triangle", "square", "pentagon", "hexagon"):
            g = geometry(shape)
            edge = np.roll(g.hole_polygon, -1, axis=0) - g.hole_polygon
            normal = np.column_stack((-edge[:, 1], edge[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            gap = (normal * (g.peg_polygon - g.hole_polygon)).sum(1)
            np.testing.assert_allclose(gap, g.clearance / 2, atol=1e-12)
        l_model = build_model(geometry("l"))
        self.assertEqual(l_model.geom("wall_notch").type, mjtGeom.mjGEOM_BOX)
        self.assertGreater(l_model.geom("wall_notch").size[0], 0)
        for clearance in (0.0001, 0.00002):
            c_geom = geometry("circle", clearance)
            self.assertLessEqual(c_geom.circle_error, clearance / 2 * 0.05)

    def test_config_and_depth_validation(self):
        self.assertEqual(asdict(InsertionCfg.from_checkpoint(asdict(InsertionCfg()))),
                         asdict(InsertionCfg()))
        invalid_cases: tuple[dict[str, Any], ...] = (
            {"decimation": 10}, {"target_depth_min_m": 0},
            {"target_depth_max_fraction": 1}, {"target_depth_min_m": 0.023},
            {"scale": 0.1}, {"fixed_target_depth_m": 0.025},
            {"circle_sides": 2}, {"circle_sides": 64.5},
            {"clearance_curriculum": True, "clearance": 0.003},
        )
        for values in invalid_cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                InsertionCfg(**values)

    def test_fixed_clearance_evaluation_of_shapes_without_adr_or_rims(self):
        cfg = InsertionCfg(shapes=("triangle",), shape_randomization=True, noise_curriculum=True,
                           noise_curriculum_mode="adr", clearance_curriculum=True, rim_probability=1.)
        for shape in ("square", "circle"):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "ADR"):
                    replace(cfg, shapes=(shape,))
                evaluation = replace(cfg, shapes=(shape,), training=False)
                self.assertEqual(evaluation.active_clearance_levels, (cfg.clearance,))
                group = PlantGroup.__new__(PlantGroup)
                group.cfg, group.device, group.n, group.difficulty = evaluation, "cpu", 1, 1.
                group.generator = torch.Generator().manual_seed(7)
                group.geometry = geometry(shape, evaluation.clearance, randomized=True)
                group.model = build_model(group.geometry)
                group.default_geom_pos = torch.as_tensor(group.model.geom_pos).float()
                group._init_shape()
                self.assertEqual(group.rim_geom_ids, [])
                group.shape_params = torch.tensor([[1., 0., 0., evaluation.clearance]])
                group.opening_radius, group.rim_height, group.rim_width = torch.zeros(3, 1).unbind()
                group.randomize_shape(torch.tensor([0]))
                self.assertEqual(group.rim_height.item(), 0.)
                data = MjData(group.model)
                data.qpos[:3] = [0., 0., 0.045]
                mj_forward(group.model, data)
                self.assertEqual(data.ncon, 0)
                data.qpos[0] += 0.002
                mj_forward(group.model, data)
                self.assertGreater(data.ncon, 0)

    def test_square_uses_boxes_with_exact_inner_clearance(self):
        for scale, clearance in ((1., 0.0001), (0.8, 0.0005), (1.2, 0.002)):
            with self.subTest(scale=scale, clearance=clearance):
                g = geometry("square", clearance, scale)
                model = build_model(g)
                self.assertEqual(model.ngeom, 5)
                self.assertEqual(model.nmesh, 0)
                self.assertTrue(np.all(model.geom_type == mjtGeom.mjGEOM_BOX))
                np.testing.assert_allclose(
                    model.geom("peg").size,
                    [g.radius/math.sqrt(2)-clearance/2]*2 + [g.peg_length/2])
                data = MjData(model)
                data.qpos[:3] = [0, 0, 0.04*scale]
                data.qpos[3:] = [1, 0, 0, 0]
                mj_forward(model, data)
                self.assertEqual(data.ncon, 0)
                for angle in np.arange(4)*math.pi/2 + math.pi/4:
                    direction = np.array([math.cos(angle), math.sin(angle)])
                    for penetration in (-2e-6, 2e-6):
                        data.qpos[:2] = direction*(clearance/2 + penetration)
                        mj_forward(model, data)
                        if penetration < 0:
                            self.assertEqual(data.ncon, 0)
                        else:
                            self.assertGreater(data.ncon, 0)
                            self.assertAlmostEqual(
                                min(c.dist for c in data.contact), -penetration, places=9)

    def test_explicit_circle_resolution(self):
        g = geometry("circle", 0.0002, circle_sides=64)
        self.assertEqual(len(g.hole_polygon), 64)
        self.assertAlmostEqual(g.circle_error, 0.01 * (1 - math.cos(math.pi / 64)))
        self.assertAlmostEqual((g.clearance / 2 - g.circle_error) * 1e6, 87.954562, places=5)
        model = build_model(g)
        self.assertAlmostEqual(model.geom("peg").size[0], 0.0099)
        data = MjData(model)
        data.qpos[:3] = [0, 0, 0.04]
        data.qpos[3:] = [1, 0, 0, 0]
        mj_forward(model, data)
        self.assertEqual(data.ncon, 0)
        with self.assertRaisesRegex(ValueError, "closes"):
            geometry("circle", 0.00002, circle_sides=64)
        with self.assertRaises(ValueError):
            geometry("circle", circle_sides=2)

    def test_quaternion_rotvec_and_tip_pivot(self):
        q = rotvec_quat(torch.tensor([[0.2, -0.4, 0.5], [0., math.pi, 0.]], dtype=torch.float64))
        torch.testing.assert_close(canonical_quat(q), canonical_quat(-q))
        v = torch.tensor([[0.1, -0.2, 0.3], [0., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(quat_rotvec(rotvec_quat(v)), v)
        p_tip = torch.tensor([[0.002, -0.001, 0.005]], dtype=torch.float64)
        q_init = rotvec_quat(torch.tensor([[0.05, -0.03, 0.1]], dtype=torch.float64))
        p_target, q_target = action_target(
            p_tip, q_init, torch.tensor([[0., 0., 0., 1., -1., 0.5]], dtype=torch.float64),
            0.001, math.radians(0.5))
        torch.testing.assert_close(p_target, p_tip)
        self.assertFalse(torch.allclose(q_target, q_init))
        p_unbounded, _ = action_target(
            p_tip, q_init, torch.tensor([[2., 0., 0., 0., 0., 0.]], dtype=torch.float64),
            0.001, math.radians(0.5))
        torch.testing.assert_close(p_unbounded, p_tip + torch.tensor(
            [[0.002, 0., 0.]], dtype=torch.float64))

    def test_wrench_transfer_and_observation_contract(self):
        q = rotvec_quat(torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.float64))
        f = torch.tensor([[3., -4., 12.]], dtype=torch.float64)
        m = torch.tensor([[0.1, -0.2, 0.05]], dtype=torch.float64)
        r = torch.tensor([[0.001, -0.002, -0.08]], dtype=torch.float64)
        shifted = transform_wrench(f, m, q, r)
        restored = transform_wrench(shifted[:, :3], shifted[:, 3:], quat_inv(q),
                                    rotate(quat_inv(q), -r))
        torch.testing.assert_close(restored, torch.cat((f, m), -1))
        obs = actor_observation(torch.zeros(2, 3), torch.tensor([[1., 0, 0, 0]] * 2),
                                torch.ones(2, 3), torch.tensor([[1., 0, 0, 0]] * 2),
                                torch.zeros(2, 6))
        self.assertEqual(obs.shape, (2, 13))

    def test_rough_surface_holes_remain_open_and_config_validates(self):
        for shape in ALL_SHAPES:
            with self.subTest(shape=shape):
                g = geometry(shape, circle_sides=64 if shape == "circle" else None,
                             plate_outer_radius_m=0.08, rough_surface=True)
                model = build_model(g)
                self.assertGreater(model.nmocap, 0)
                data = MjData(model)
                data.qpos[:3] = [0, 0, 0.04]
                data.qpos[3:] = [1, 0, 0, 0]
                mj_forward(model, data)
                self.assertEqual(data.ncon, 0)
        for bad in (
            {"enable_timeout": "no"},
            {"rough_surface": 1},
            {"rough_max_amplitude_m": -0.001},
            {"rough_wavelength_min_m": 0.02, "rough_wavelength_max_m": 0.01},
            {"rough_surface": True, "rough_max_amplitude_m": 0.025, "initial_z_max": 0.02},
            {"stagnation_cost_per_second": -0.1},
            {"stagnation_speed_threshold_mps": 0.0},
            {"rotation_bias": -0.01},
            {"grasp_z_error": -0.001},
            {"hole_depth_m": 0.040},
            {"align_goal_z_to_observation": "yes"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                InsertionCfg(**bad)


if __name__ == "__main__":
    unittest.main()
