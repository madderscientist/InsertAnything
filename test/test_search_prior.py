"""孔位搜索：形状随机化、重置、不准确目标点、循环模型与 gSDE"""

from argparse import Namespace
from dataclasses import asdict, replace
import math
from pathlib import Path
import tempfile
import unittest

from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
import torch
from torch.distributions import Normal
import warp as wp

from evaluate import load_checkpoint
from export_policy import export, export_onnx
from insertion import InsertionCfg, InsertionEnv
from insertion.env import warp_stream
from insertion.model import CHRONO_MAX_STEPS, GSDE_DISTRIBUTION, MinGRU, SkipRNNModel
from insertion.ppo import SmoothPPO
from insertion.transforms import rotate, tip_pose
from search_prior.train_search import SEARCH_SHAPES, search_runner
from train import ContinuousRunner, atomic_json, contract, resume_training

SEARCH = dict(initial_xy=0.005, workspace_xy_radius_m=0.025)


class SearchPriorTests(unittest.TestCase):
    def test_randomized_shapes_insert_only_when_aligned(self):
        cfg = InsertionCfg(shapes=SEARCH_SHAPES, envs_per_shape=16, shape_randomization=True,
                           rim_probability=0.5, training=False, dynamics="nominal", seed=3,
                           grasp_position_error=0., grasp_z_error=0., tip_calibration_error=0.,
                           stiction_force=0., stiction_moment=0., enable_timeout=False,
                           physics_cuda_graph=True, observation_cuda_graph=True)
        env = InsertionEnv(cfg, "cuda:0")
        for g in env.groups:
            with self.subTest(shape=g.shape):
                n = g.n
                shifted = torch.arange(n, device=g.device) >= n // 2
                # 单孔偏移半个孔口半径；双脚偏移一个槽距，一只脚正对中间干扰槽
                offset = g.shape_params[:, 2] if g.shape == "dual" else 0.5 * g.opening_radius
                self.assertGreater(float(g.shape_params[:, 0].std()), 0.)
                with torch.inference_mode():
                    pos = torch.zeros(n, 3, device=g.device)
                    pos[:, 0] = torch.where(shifted, offset, 0.)
                    pos[:, 2] = 0.002
                    q = torch.tensor([1., 0, 0, 0], device=g.device).expand(n, 4)
                    g.sim.data.qpos[:, :3] = pos - rotate(q, g.offset)
                    g.sim.data.qpos[:, 3:7] = q
                    g.sim.data.qvel[:] = 0
                    g.sim.forward()
                    action = torch.zeros(n, 6, device=g.device)
                    action[:, 2] = -1.
                    for _ in range(40):
                        g.step(action)
                    q = g.sim.data.qpos[:, 3:7]
                    tip = g.sim.data.qpos[:, :3] + rotate(q, g.offset)
                    inserted = g.geometry.definition.inserted(g.geo, g.shape_params, tip, q)
                self.assertTrue(bool((tip[~shifted, 2] < -0.01).all()))
                self.assertTrue(bool((tip[shifted, 2] > -(g.rim_height[shifted] + 0.0006)).all()))
                self.assertTrue(bool(inserted[~shifted].all()))
                if g.shape == "dual":
                    self.assertFalse(bool(inserted[shifted].any()))

    def test_domain_randomization_grows_with_curriculum_difficulty(self):
        cfg = InsertionCfg(shapes=("rectangle",), envs_per_shape=256, shape_randomization=True,
                           rim_probability=1.0, target_lowpass=True, target_lowpass_off_probability=1.0,
                           mass_min_kg=0.08, seed=2)
        g = InsertionEnv(cfg, "cuda:0").groups[0]
        ids = torch.arange(g.n, device=g.device)
        g.difficulty = 0.
        tip_z = lambda: tip_pose(*g.state()[:2], g.offset)[:, 2]
        g.reset(ids)
        torch.testing.assert_close(tip_z(), torch.full((g.n,), cfg.initial_z_min, device=g.device))
        torch.testing.assert_close(g.damping_ratio, torch.ones_like(g.damping_ratio))
        self.assertTrue(bool((g.rim_height == 0).all()))
        self.assertTrue(bool(g.lowpass_off.all()))
        self.assertLess(float(g.apparent_mass.std()), 1e-6)
        g.difficulty = 1.
        g.reset(ids)
        self.assertGreater(float(tip_z().max()), 0.045)
        self.assertLess(float(g.damping_ratio.min()), 0.4)
        self.assertTrue(bool((g.rim_height > 0).all()))

    def test_masked_graph_reset_keeps_running_envs_and_matches_eager(self):
        for shape in ("dual", "pentagon"):
            with self.subTest(shape=shape):
                self.check_masked_graph_reset(shape)

    def check_masked_graph_reset(self, shape: str) -> None:
        cfg = replace(InsertionCfg(shapes=(shape,), envs_per_shape=32, shape_randomization=True,
                                   rim_probability=0.5, hard_state_reset_probability=0.5, training=True,
                                   noise_curriculum=True, noise_curriculum_mode="adr", clearance_curriculum=True,
                                   physics_cuda_graph=True, observation_cuda_graph=True, seed=4), **SEARCH)
        env = InsertionEnv(cfg, "cuda:0")
        g = env.groups[0]
        g.difficulty = 0.7
        g.set_clearance(0.0005)
        g.noise_scale, g.noise_floor, g.clearance_max = 0.8, 0.3, 0.001
        g.hard_valid[:16] = True

        def state():
            tensors = {k: v.clone() for k, v in vars(g).items() if isinstance(v, torch.Tensor)}
            # 重置后整组 forward 会按当前 qpos 刷新派生量（与原即时重置相同），故只比较状态量
            tensors.update({k: getattr(g.sim.data, k)[:].clone() for k in ("qpos", "qvel", "xfrc_applied")})
            tensors.update({k: getattr(g.sim.model, k)[:].clone() for k in g.expanded_fields})
            return tensors

        with torch.inference_mode():
            for _ in range(3):
                env.step(torch.full((g.n, 6), 0.2, device=g.device))
            mask = torch.zeros(g.n, dtype=torch.bool, device=g.device)
            mask[[2, 9, 31]] = True
            for _ in range(3):
                before = state()
                g.reset_masked(mask)
                after = state()
                for key, old in before.items():
                    new = after[key]
                    if key in ("targets_p", "targets_q"):
                        old, new = old.transpose(0, 1), new.transpose(0, 1)
                    if old.dim() and old.shape[0] == g.n:
                        self.assertTrue(torch.equal(old[~mask], new[~mask]), key)
                self.assertTrue(bool((g.length[mask] == 0).all()))
            self.assertIsNotNone(g.masked_reset_graph)
            # 同一发生器状态下，即时执行与图回放逐位一致
            rng = g.generator.get_state()
            signature = g.masked_reset_signature
            g.masked_reset_signature = None
            g.reset_masked(mask)
            eager = state()
            g.generator.set_state(rng)
            g.masked_reset_signature = signature
            g.reset_masked(mask)
            replay = state()
        for key, value in eager.items():
            self.assertTrue(torch.equal(value, replay[key]), key)

    def test_hard_start_restores_recorded_shape(self):
        cfg = InsertionCfg(shapes=("rectangle",), envs_per_shape=8, shape_randomization=True,
                           hard_state_reset_probability=1.0, seed=5)
        g = InsertionEnv(cfg, "cuda:0").groups[0]
        g.difficulty = 1.
        params = torch.tensor([0.005, 0.002, 0., 0.0004], device=g.device)
        g.hard_params[:] = params
        g.hard_rim[:] = torch.tensor([0.0015, 0.003], device=g.device)
        g.hard_pos[:] = torch.tensor([0.01, 0., 0.003], device=g.device)
        g.hard_valid[:] = True
        ids = torch.arange(g.n, device=g.device)
        g.reset(ids)
        self.assertTrue(bool(g.hard_start.all()))
        torch.testing.assert_close(g.shape_params, params.expand(g.n, -1))
        torch.testing.assert_close(g.rim_height, torch.full((g.n,), 0.0015, device=g.device))
        peg = g.geom_index["peg"]
        torch.testing.assert_close(g.sim.model.geom_size[:, peg, :2],
                                   (params[:2] - params[3]/2).expand(g.n, -1))
        wall = g.geom_index["wall_left"]
        self.assertTrue(bool(torch.allclose(g.sim.model.geom_pos[:, wall, 2],
                                            torch.full((g.n,), -g.geometry.depth/2 - 0.0015, device=g.device))))

    def test_adr_clearance_offsets_polygon_walls_without_gaps(self):
        cfg = InsertionCfg(shapes=("pentagon", "rectangle"), envs_per_shape=256, shape_randomization=True,
                           training=True, noise_curriculum=True, noise_curriculum_mode="adr",
                           clearance_curriculum=True, seed=6)
        env = InsertionEnv(cfg, "cuda:0")
        for g in env.groups:
            g.set_clearance(0.0005)
            g.reset(torch.arange(g.n, device=g.device))
            c = g.shape_params[:, 3]
            self.assertEqual(g.clearance, cfg.clearance)
            self.assertTrue(bool(((c >= 0.0005) & (c <= 0.002 + 1e-9)).all()))
            probe = g.episode_boundary == 2
            self.assertTrue(bool(probe.any()) and bool((c[probe] == 0.0005).all()))
            # 收拢阶段：间隙上限与噪声下限向难侧移动
            g.noise_scale, g.noise_floor, g.clearance_max = 0.8, 0.5, 0.001
            g.reset(torch.arange(g.n, device=g.device))
            c, s = g.shape_params[:, 3], g.episode_noise_scale
            self.assertTrue(bool(((c >= 0.0005) & (c <= 0.001 + 1e-9)).all()))
            self.assertTrue(bool(((s >= 0.5) & (s <= 0.8 + 1e-6)).all()))
            self.assertTrue(bool(torch.allclose(s[g.episode_boundary == 1], torch.tensor(0.8, device=g.device))))
            g.noise_floor, g.clearance_max = 0., 0.002
            g.reset(torch.arange(g.n, device=g.device))
        g, sides = env.groups[0], 5
        half_angle = math.pi/sides
        peg_apothem = g.geometry.radius*g.shape_params[:, 0]*math.cos(half_angle) - cfg.clearance/2
        angles = torch.arange(sides, device=g.device)*2*half_angle + math.pi/2
        vertices = torch.stack((angles.cos(), angles.sin()), -1)
        walls = []
        for i in range(sides):
            gid = g.geom_index[f"wall_{i}"]
            center, half = g.sim.model.geom_pos[:, gid, :2], g.sim.model.geom_size[:, gid, :2]
            w, _, _, z = g.model.geom_quat[gid]
            walls.append((center, half, 2*math.atan2(z, w)))
            # 孔壁内表面到插头对应侧面的距离为半个回合间隙
            torch.testing.assert_close(torch.linalg.vector_norm(center, dim=-1) - half[:, 1] - peg_apothem,
                                       g.shape_params[:, 3]/2)
        # 孔口顶点外侧 0.05 mm 落在孔壁内，内侧 0.05 mm 不在任何孔壁内
        for sign, expected in ((1., True), (-1., False)):
            points = (g.opening_radius[:, None, None] + sign*5e-5)*vertices
            covered = torch.zeros(g.n, sides, dtype=torch.bool, device=g.device)
            for center, half, yaw in walls:
                d = points - center[:, None]
                local = torch.stack((d[..., 0]*math.cos(yaw) + d[..., 1]*math.sin(yaw),
                                     -d[..., 0]*math.sin(yaw) + d[..., 1]*math.cos(yaw)), -1)
                covered |= (local.abs() <= half[:, None]).all(-1)
            self.assertTrue(bool((covered == expected).all()), sign)

    def test_recurrent_actor_critic_trains_resumes_and_exports(self):
        torch.set_num_threads(1)
        cfg = InsertionCfg(envs_per_shape=4, shapes=("square",), episode_seconds=0.08, **SEARCH)
        config = search_runner(Namespace(steps=4, rnn_hidden=64, rnn_layers=None, mini_batches=4,
                                         gsde_resample_steps=25, phase_coef=0.1, phase_lr=1e-3))
        self.assertEqual(config["actor"]["rnn_num_layers"], 2)
        self.assertEqual(config["critic"]["rnn_num_layers"], 2)
        self.assertEqual(config["algorithm"]["actor_min_lr"], 1e-6)
        self.assertEqual(config["actor"]["distribution_cfg"]["class_name"], GSDE_DISTRIBUTION)
        metadata = {"environment": asdict(cfg), "runner": config, "contract": contract(cfg)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_json(output / "config.json", metadata)
            runner = ContinuousRunner(InsertionEnv(cfg, "cuda:0"), config, output, metadata)
            for model in (runner.alg.actor, runner.alg.critic):
                self.assertIsInstance(model.rnn.rnn, MinGRU)
            # 初始保留比例 u/(1+u)，u 铺满 [1, T-1]
            retain = 1 - torch.sigmoid(runner.alg.actor.rnn.rnn.layers[0].bias[:64].detach())
            self.assertGreaterEqual(float(retain.min()), 0.5 - 1e-6)
            self.assertLessEqual(float(retain.max()), 1 - 1 / CHRONO_MAX_STEPS + 1e-6)
            tau = 1 / (1 - retain)
            self.assertLess(float(tau.min()), 50.)
            self.assertGreater(float(tau.max()), 400.)
            before = runner.alg.critic.rnn.rnn.layers[1].weight.detach().clone()
            runner.learn(1)
            self.assertFalse(torch.equal(before, runner.alg.critic.rnn.rnn.layers[1].weight.detach()))
            saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(saved["critic_state_dict"]["phase_head.weight"].shape, (4, 64))
            labels = runner.env.get_observations()["phase"]
            self.assertEqual(labels.dtype, torch.long)
            self.assertTrue(bool(((labels >= 0) & (labels < 4)).all()))
            resumed = ContinuousRunner(InsertionEnv(cfg, "cuda:0"), config, output, metadata)
            resume_training(resumed, cfg, 1)
            torch.testing.assert_close(resumed.alg.critic.rnn.rnn.layers[1].weight.cpu(),
                                       saved["critic_state_dict"]["rnn.rnn.layers.1.weight"])
            # 旧检查点 actor/critic 共用一组：按当前分组切分，critic 恢复与 actor 相同的学习率
            optimizer = saved["optimizer_state_dict"]
            merged = {**optimizer, "param_groups": [{**optimizer["param_groups"][0], "lr": 1e-6, "params": [
                i for group in optimizer["param_groups"] for i in group["params"]]}]}
            resumed.alg.load({**saved, "optimizer_state_dict": merged}, None, True)
            self.assertEqual([g["lr"] for g in resumed.alg.optimizer.param_groups],
                             [1e-6, 1e-6, 1e-3])
            resumed.learn(1)
            _, actor = load_checkpoint(output / "checkpoint.pt", "cuda:0")
            self.assertIsInstance(actor.rnn.rnn, MinGRU)
            self.assertEqual(actor.rnn.rnn.hidden_size, 64)
            self.assertEqual(actor.obs_dim, 13)
            self.assertFalse(any(isinstance(m, torch.nn.LayerNorm) for m in actor.modules()))
            # 导出函数内部逐步校验 TorchScript/ONNX 与原模型的序列数值等价
            export(output / "checkpoint.pt", output / "jit", samples=100)
            export_onnx(output / "checkpoint.pt", output / "onnx", samples=100)

    def test_mingru_update_batch_matches_rollout_across_resets(self):
        # 更新时按回合切分、零初始化的整段计算，须与带异步复位的逐步 rollout 一致（复位后无残留记忆）
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                torch.manual_seed(5)
                n, steps = 6, 12
                obs = TensorDict({"actor": torch.randn(n, 13), "critic": torch.randn(n, 15)}, [n], device=device)
                groups = {"actor": ["actor"], "critic": ["critic"]}
                model = dict(hidden_dims=(8,), rnn_type="gru", rnn_hidden_dim=8, rnn_num_layers=2)
                actor = SkipRNNModel(obs, groups, "actor", 6, **model,
                                     distribution_cfg={"class_name": GSDE_DISTRIBUTION}).to(device)
                critic = SkipRNNModel(obs, groups, "critic", 1, **model).to(device)
                storage = RolloutStorage("rl", n, steps, obs, [6], device=device)
                alg = SmoothPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1, device=device)
                dones = torch.rand(steps, n, device=device) < 0.25
                with torch.inference_mode():
                    for t in range(steps):
                        step_obs = obs.apply(torch.randn_like)
                        alg.act(step_obs)
                        alg.process_env_step(step_obs, torch.randn(n, device=device), dones[t].long(), {})
                    alg.compute_returns(obs)
                    batch = next(storage.recurrent_mini_batch_generator(1, 1))
                    actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                          stochastic_output=True)
                for new, old in zip(actor.output_distribution_params, batch.old_distribution_params, strict=True):
                    torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-5)

    def test_mingru_fused_scan_gradients_match_stepwise(self):
        torch.manual_seed(6)
        rnn = MinGRU(13, 32, 2, CHRONO_MAX_STEPS).cuda()
        x, h0 = torch.randn(64, 5, 13, device="cuda"), torch.randn(2, 5, 32, device="cuda").tanh()
        out, last = rnn(x, h0)
        fused = torch.autograd.grad(out.square().sum() + last.sum(), list(rnn.parameters()))
        h, steps = h0, []
        for t in range(x.shape[0]):
            y, h = rnn(x[t:t + 1], h)
            steps.append(y)
        stepwise = torch.autograd.grad(torch.cat(steps).square().sum() + h.sum(), list(rnn.parameters()))
        torch.testing.assert_close(out, torch.cat(steps), atol=1e-6, rtol=1e-5)
        for a, b in zip(fused, stepwise, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)

    def test_ppo_single_environment_minibatch_pairs_likelihoods_by_step(self):
        torch.manual_seed(8)
        obs = TensorDict({"actor": torch.randn(1, 13), "critic": torch.randn(1, 15)}, [1])
        groups = {"actor": ["actor"], "critic": ["critic"]}
        model = dict(hidden_dims=(8,), rnn_type="gru", rnn_hidden_dim=8, rnn_num_layers=1)
        actor = SkipRNNModel(obs, groups, "actor", 6, **model,
                             distribution_cfg={"class_name": GSDE_DISTRIBUTION, "resample_steps": 1})
        critic = SkipRNNModel(obs, groups, "critic", 1, **model)
        storage = RolloutStorage("rl", 1, 4, obs, [6])
        alg = SmoothPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1)
        with torch.inference_mode():
            for _ in range(4):
                alg.act(obs)
                alg.process_env_step(obs, torch.zeros(1), torch.zeros(1, dtype=torch.long), {})
            alg.compute_returns(obs)
            storage.advantages.fill_(1)
        self.assertAlmostEqual(alg.update()["surrogate"], -1.0, places=5)

    def test_critic_rate_on_resume_and_kl_update(self):
        torch.manual_seed(7)
        obs = TensorDict({"actor": torch.randn(4, 13), "critic": torch.randn(4, 15)}, [4])
        groups = {"actor": ["actor"], "critic": ["critic"]}
        model = dict(hidden_dims=(8,), rnn_type="gru", rnn_hidden_dim=8)
        for schedule in ("actor", 3e-5):
            with self.subTest(schedule=schedule):
                actor = SkipRNNModel(obs, groups, "actor", 6, **model, rnn_num_layers=1,
                                     distribution_cfg={"class_name": GSDE_DISTRIBUTION})
                critic = SkipRNNModel(obs, groups, "critic", 1, **model, rnn_num_layers=2)
                storage = RolloutStorage("rl", 4, 4, obs, [6])
                alg = SmoothPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
                                learning_rate=1e-4, critic_learning_rate=schedule)
                saved = alg.save()
                saved["optimizer_state_dict"]["param_groups"][0]["lr"] = 6e-5
                alg.load(saved, None, True)
                critic_rate = 6e-5 if schedule == "actor" else schedule
                self.assertEqual([g["lr"] for g in alg.optimizer.param_groups], [6e-5, critic_rate])
                with torch.inference_mode():
                    for _ in range(4):
                        alg.act(obs)
                        alg.process_env_step(obs, torch.randn(4), torch.zeros(4, dtype=torch.long), {})
                    alg.compute_returns(obs)
                    storage.distribution_params[0].add_(1.0)  # Force KL above the decrease threshold.
                alg.update()
                actor_rate, critic_rate = [g["lr"] for g in alg.optimizer.param_groups]
                self.assertAlmostEqual(actor_rate, 4e-5)
                self.assertEqual(critic_rate, actor_rate if schedule == "actor" else schedule)

    def test_gsde_noise_is_state_dependent_consistent_and_matches_likelihood(self):
        torch.manual_seed(0)
        obs = TensorDict({"actor": torch.randn(4096, 13)}, [4096])
        model = SkipRNNModel(obs, {"actor": ["actor"]}, "actor", 6, hidden_dims=(32, 16),
                             distribution_cfg={"class_name": GSDE_DISTRIBUTION, "init_std": 0.4,
                                               "std_range": (0.02, 1.0), "resample_steps": 3},
                             rnn_type="gru", rnn_hidden_dim=16, rnn_num_layers=1)
        dist = model.distribution
        with torch.no_grad():
            first = model(obs, stochastic_output=True)
            mean, std = dist.mean, dist.std
            torch.testing.assert_close(std, torch.full_like(std, 0.4))
            # 探索矩阵保持期间，同一状态得到同一扰动；只复位 GRU 隐状态
            model.rnn.reset()
            second = model(obs, stochastic_output=True)
            torch.testing.assert_close(second, first)
            noise = (first - mean) / std
            self.assertLess(abs(float(noise.mean())), 0.05)
            self.assertLess(abs(float(noise.std()) - 1.), 0.05)
            torch.testing.assert_close(model.get_output_log_prob(first),
                                       Normal(mean, std).log_prob(first).sum(-1))
            # 结束的环境重抽探索矩阵，其余不变
            dones = torch.zeros(4096, dtype=torch.bool)
            dones[:8] = True
            epsilon = dist.epsilon.clone()
            model.reset(dones)
            self.assertFalse(torch.equal(dist.epsilon[:8], epsilon[:8]))
            torch.testing.assert_close(dist.epsilon[8:], epsilon[8:])
            # 每 resample_steps 次采样整体重抽
            model(obs, stochastic_output=True)
            epsilon = dist.epsilon.clone()
            model(obs, stochastic_output=True)
            self.assertFalse(torch.equal(dist.epsilon[8:], epsilon[8:]))

    def test_concurrent_group_streams_match_serial(self):
        cfg = InsertionCfg(envs_per_shape=8, shapes=("square", "triangle", "l"), seed=7,
                           physics_cuda_graph=True, observation_cuda_graph=True, **SEARCH)
        generator = torch.Generator(device="cuda:0").manual_seed(3)
        actions = [torch.randn(24, 6, generator=generator, device="cuda:0") for _ in range(40)]
        outputs = []
        for concurrent in (False, True):
            env = InsertionEnv(cfg, "cuda:0")
            if not concurrent:
                env.launch_groups = lambda chunks, env=env: [
                    g.launch(a) for g, a in zip(env.groups, chunks, strict=True)]
            with torch.inference_mode():
                trace = [env.step(a)[0]["critic"].clone() for a in actions]
            self.assertTrue(all(g.physics_rollout_graph is not None for g in env.groups))
            # 即时重置在组流上分配 Warp 临时数组，只有阻塞流能与 Warp 的空流异步释放排序
            self.assertTrue(all(warp_stream(s).is_blocking for s in env.group_streams))
            outputs.append(torch.stack(trace))
        torch.testing.assert_close(outputs[1], outputs[0])


if __name__ == "__main__":
    unittest.main()
