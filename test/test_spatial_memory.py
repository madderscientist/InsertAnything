"""Local spatial writes, height targets and recurrent PPO replay on CPU."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import mujoco
import mujoco_warp as mjwarp
import numpy as np
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
import torch
import warp as wp

from evaluate import load_checkpoint
from insertion.env import PlantGroup
from insertion.geometry import build_model, geometry
from insertion.model import GSDE_DISTRIBUTION, SKIP_RNN_MODEL, SkipRNNModel
from insertion.ppo import SmoothPPO
from insertion.spatial_memory import (BLOCK, CELLS, CHANNELS, COARSE, GRID, HEIGHT_TARGET,
                                      REPLAY, SpatialMemory, local_patch)
from insertion.surface_height import SurfaceRays


class SpatialMemoryTests(unittest.TestCase):
    def test_local_writes_outside_grid_masked_supervision_and_partial_reset(self):
        torch.manual_seed(10)
        m = SpatialMemory(5)
        xy = torch.tensor([[0.0002, -0.0011], [0.0238, 0.0238], [0.06, 0.]])
        latent = torch.randn(3, 5)
        cells, _, valid = local_patch(xy)
        m(latent, xy)
        touched = torch.zeros(3, GRID**2, dtype=torch.bool)
        touched.scatter_(1, cells, valid)
        self.assertEqual(m.grid.shape, (3, CHANNELS, GRID, GRID))
        self.assertTrue(torch.equal(m.flat_grid.ne(0).any(-1), touched))
        self.assertEqual(touched.sum(1).tolist(), [9, 4, 0])
        predicted, marked = m.decoded_surface(0)
        self.assertTrue(torch.equal(marked.flatten(), touched[0]))
        self.assertFalse(predicted[~marked].any())
        m(latent.unsqueeze(0), xy.unsqueeze(0), (torch.zeros_like(m.flat_grid), torch.zeros_like(m.coarse),
                                                  torch.zeros(1, 3, dtype=torch.bool)))
        target = torch.zeros(1, 3, CELLS)
        target[0, ~valid] = 100
        torch.testing.assert_close(m.height_loss(target)[0], m.height_loss(torch.zeros_like(target))[0])
        original = m.grid.clone()
        m.reset(torch.tensor([False, True, False]))
        torch.testing.assert_close(m.grid[0], original[0])
        self.assertFalse(m.grid[1:].any())
        self.assertFalse(m.coarse[1:].any())

    def test_sparse_replay_matches_dense_recurrence_and_gradients(self):
        torch.manual_seed(17)
        steps, n = 11, 3
        m = SpatialMemory(4)
        ref = copy.deepcopy(m)
        latent = torch.randn(steps, n, 4, requires_grad=True)
        other = latent.detach().clone().requires_grad_()
        xy = torch.randn(steps, n, 2) * 0.001
        xy[4, :, 0] = 0.07
        xy[:, 1, 0] += 0.023
        dones = torch.zeros(steps, n, dtype=torch.bool)
        dones[[1, 4, 5], 0] = True
        dones[6, 2] = True
        grid = torch.randn(n, GRID**2, CHANNELS) * 0.1

        def pooled(h):
            return h.reshape(n, COARSE, BLOCK, COARSE, BLOCK, CHANNELS).sum((2, 4)).reshape(n, COARSE**2, CHANNELS)

        features = m(latent, xy, (grid, pooled(grid), dones))
        grad = torch.randn_like(features)
        (grad * features).sum().backward()
        reference = []
        for step in range(steps):
            if step:
                grid = grid * (~dones[step - 1])[:, None, None]
            cell, offset, valid = local_patch(xy[step])
            evidence = ref.evidence(other[step])[:, None] + ref.offset(offset.clamp(-3, 3))
            gate, candidate = ref.write(torch.nn.functional.elu(evidence)).chunk(2, -1)
            index = cell[..., None].expand(-1, -1, CHANNELS)
            old = grid.gather(1, index)
            written = torch.ones_like(gate[..., :1])
            gate = torch.cat((gate.sigmoid(), written), -1)
            candidate = torch.cat((candidate.tanh(), written), -1)
            new = old + gate * valid[..., None] * (candidate - old)
            grid = grid.scatter(1, index, new)
            read = torch.cat(((new * valid[..., None]).flatten(1), (pooled(grid) / BLOCK).flatten(1),
                              (xy[step] / 0.024).clamp(-2, 2)), -1)
            reference.append(ref.read(read))
        reference = torch.stack(reference)
        (grad * reference).sum().backward()
        torch.testing.assert_close(features, reference, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(latent.grad, other.grad, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(m.write.weight.grad, ref.write.weight.grad, rtol=2e-5, atol=3e-6)

    def test_height_rays_match_mujoco_and_exclude_tool(self):
        for shape in ("triangle", "pentagon", "hexagon", "l", "rectangle", "dual"):
            with self.subTest(shape=shape), wp.ScopedDevice("cpu"):
                geom = geometry(shape, 0.0002, 1., None, 0.06, False, 0.022, True)
                model = build_model(geom)
                # Lower the surrounding board by 1.5 mm and put one rim box at the mouth.
                tool = model.body("tool").id
                rim = model.geom("rim_0").id
                for i in range(model.ngeom):
                    if model.geom_bodyid[i] == model.body("hole").id and not model.geom(i).name.startswith("rim_"):
                        model.geom_pos[i, 2] -= 0.0015
                model.geom_pos[rim] = (0.012, 0, -0.00075)
                model.geom_size[rim] = (0.001, 0.003, 0.00075)
                model.geom_quat[rim] = (1, 0, 0, 0)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                coords = torch.tensor([[-18., -6.35, -3., 0., 3., 6.35, 10., 12., 18.]]) * 0.001
                xy = torch.stack((coords, torch.zeros_like(coords)), dim=-1)
                rays = SurfaceRays(1, "cpu", tool, 0.08)
                result = rays.sample(mjwarp.put_model(model), mjwarp.put_data(model, data, nworld=1),
                                     xy, torch.tensor([[-0.0015]]), 0.022)
                expected = []
                for point in xy[0].numpy():
                    hit = mujoco.mj_ray(model, data, np.array([*point, 0.08]), np.array([0., 0., -1.]),
                                        None, True, tool, np.zeros(1, dtype=np.int32))
                    expected.append(max(0.08 - hit + 0.0015, -0.022) if hit >= 0 else -0.022)
                torch.testing.assert_close(result[0], torch.tensor(expected).float(), rtol=0., atol=2e-6)
                self.assertAlmostEqual(result[0, 0].item(), 0., places=5)
                self.assertAlmostEqual(result[0, 7].item(), 0.0015, places=5)
                if shape == "dual":
                    for slot in (1, 3, 5):
                        self.assertAlmostEqual(result[0, slot].item(), -0.022, places=5)

    def test_training_label_projects_observed_cells_into_the_true_hole_frame(self):
        with wp.ScopedDevice("cpu"):
            g = PlantGroup.__new__(PlantGroup)
            g.n, g.device = 2, "cpu"
            g.geometry = geometry("rectangle", 0.0002, 1., None, 0.06, False, 0.025, True)
            model = build_model(g.geometry)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            g.sim = SimpleNamespace(wp_model=mjwarp.put_model(model), wp_data=mjwarp.put_data(model, data, nworld=2))
            g.body_id = model.body("tool").id
            g.surface_rays, g.warp_on_current_stream = None, nullcontext
            g.cfg = SimpleNamespace(workspace_z_max_m=0.07, rough_max_amplitude_m=0., scale=1.)
            g.rim_height = torch.zeros(2)
            g.nominal_offset = g.offset = g.calibration = torch.zeros(2, 3)
            g.bias = torch.tensor([[0., 0., 0.], [0.018, 0., 0.]])
            g.state = lambda: (torch.zeros(2, 3), torch.tensor([[1., 0., 0., 0.]]).expand(2, 4), None, None)
            # Same public tip location in each map. In the second world the true hole is 18 mm to the left.
            result = g.spatial_height_target(torch.zeros(2, 13))
            torch.testing.assert_close(result[0], torch.full((CELLS,), -0.025))
            torch.testing.assert_close(result[1], torch.zeros(CELLS), rtol=0., atol=2e-6)

    def test_ppo_replay_update_checkpoint_and_label_free_inference(self):
        torch.manual_seed(31)
        n, steps = 4, 6
        groups = {"actor": ["actor"], "critic": ["critic"]}
        actor_cfg = dict(rnn_type="gru", rnn_hidden_dim=16, rnn_num_layers=1, hidden_dims=(16,),
                         spatial_memory=True, distribution_cfg={"class_name": GSDE_DISTRIBUTION})

        def observations():
            public = torch.randn(n, 13)
            public[:, :2] *= 0.002
            return TensorDict({"actor": public, "critic": torch.randn(n, 19), "free_air": torch.ones(n, 1),
                               "noise_scale": torch.ones(n, 1), "phase": torch.zeros(n, 1, dtype=torch.long),
                               HEIGHT_TARGET: -torch.rand(n, CELLS) * 0.022}, [n])

        obs = observations()
        actor = SkipRNNModel(obs, groups, "actor", 6, **copy.deepcopy(actor_cfg))
        critic = SkipRNNModel(obs, groups, "critic", 1, hidden_dims=(16,), rnn_hidden_dim=12)
        storage = RolloutStorage("rl", n, steps, obs, [6])
        ppo = SmoothPPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, phase_coef=0.1,
                        smooth_coef=0.01, height_coef=0.5)
        for _ in range(2):
            with torch.inference_mode():
                for step in range(steps):
                    ppo.act(obs)
                    obs = observations()
                    ppo.process_env_step(obs, torch.randn(n), torch.tensor([step == 2, False, step == 4, False]), {})
                ppo.compute_returns(obs)
            # Same rollout observations and policy must reproduce each stored action likelihood.
            for k, batch in enumerate(storage.recurrent_mini_batch_generator(2, 1)):
                envs = slice(k * 2, (k + 1) * 2)
                grid, coarse = ppo.spatial_start
                batch.observations.set_non_tensor(REPLAY, (grid[envs], coarse[envs], storage.dones[:, envs, 0].bool()))
                actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
                torch.testing.assert_close(actor.get_output_log_prob(batch.actions), batch.old_actions_log_prob[..., 0],
                                           rtol=1e-5, atol=1e-5)
            before = [p.detach().clone() for p in (actor.spatial.write.weight, actor.spatial.height.weight)]
            metrics = ppo.update()
            self.assertTrue(0 < metrics["height_mae_mm"] < 100)
            self.assertTrue(all(not torch.equal(p, q) for p, q in zip(before, (actor.spatial.write.weight,
                                                                                actor.spatial.height.weight))))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint.pt"
            saved = ppo.save()
            saved["infos"] = {"contract": {"version": 1}, "runner": {"actor": {**actor_cfg, "class_name": SKIP_RNN_MODEL}}}
            torch.save(saved, ckpt)
            _, restored = load_checkpoint(ckpt, "cpu")
            only_public = obs.select("actor")
            actor.reset()
            with torch.inference_mode():
                for _ in range(3):
                    torch.testing.assert_close(actor(obs), restored(only_public))
                restored.reset()
                self.assertIsNone(restored.spatial.grid)
