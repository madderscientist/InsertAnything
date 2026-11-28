"""循环 PPO + 空中力旋量观测平滑正则：抑制策略对悬空时自身惯性/振动力波动的反应"""

import math
from typing import Literal

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.utils import unpad_trajectories

from .search_map import ERASE_MAX, MAP_BATCH, MAP_KEYS, MAP_NLL, MapHead, erase_nll

WRENCH_SLICE = slice(7, 13)
SMOOTH_PPO = "insertion.ppo:SmoothPPO"
DEFAULT_ACTOR_MIN_LR = 1e-5
PHASES = 4

# TF32 下 rollout 逐步推理与更新时整段推理的误差经 GRU 累积，旧策略概率失真
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class SmoothPPO(PPO):
    def __init__(self, actor, critic, *args, smooth_coef: float = 0.0, smooth_force_std: float = 0.5,
                 smooth_moment_std: float = 0.05, phase_coef: float = 0.0, phase_learning_rate: float = 1e-3,
                 map_coef: float = 0.0, map_learning_rate: float = 1e-3,
                 actor_min_lr: float = DEFAULT_ACTOR_MIN_LR, critic_learning_rate: float | Literal["actor"] = "actor",
                 recurrent_norm_max: float | None = None, **kwargs) -> None:
        if not 0 < actor_min_lr <= 1e-2:
            raise ValueError("actor_min_lr must be in (0, 1e-2]")
        if recurrent_norm_max is not None and not 0 < recurrent_norm_max < math.inf:
            raise ValueError("recurrent_norm_max must be positive and finite")
        follows_actor = critic_learning_rate == "actor"
        if not follows_actor and (not isinstance(critic_learning_rate, (int, float))
                                  or not 0 < critic_learning_rate < math.inf):
            raise ValueError("critic_learning_rate must be positive and finite or 'actor'")
        if phase_coef > 0:
            # 线性读出 actor GRU 输出；挂在 critic 上：随 critic 保存与多卡同步，且不进入导出的 actor
            critic.phase_head = nn.Linear(actor.rnn.rnn.hidden_size, PHASES)
        if map_coef > 0:
            critic.map_head = MapHead(actor.latent_dim)
        super().__init__(actor, critic, *args, **kwargs)
        if self.rnd or self.symmetry:
            raise ValueError("SmoothPPO does not support RND or symmetry extensions")
        self.smooth_coef = smooth_coef
        self.actor_min_lr = actor_min_lr
        self.recurrent_norm_max = recurrent_norm_max
        critic_lr = self.learning_rate if follows_actor else critic_learning_rate
        self.smooth_std = torch.tensor([smooth_force_std]*3 + [smooth_moment_std]*3, device=self.device)
        self.phase_coef = phase_coef
        self.map_coef = map_coef
        self.search_map = None
        self.memory: torch.Tensor | None = None
        phase_params = list(self.critic.phase_head.parameters()) if phase_coef > 0 else []
        map_params = list(self.critic.map_head.parameters()) if map_coef > 0 else []
        aux_params = set(phase_params + map_params)
        groups = [{"params": list(self.actor.parameters())},
                  {"params": [p for p in self.critic.parameters() if p not in aux_params], "lr": critic_lr}]
        if phase_params:
            groups.append({"params": phase_params, "lr": phase_learning_rate})
        if map_params:
            groups.append({"params": map_params, "lr": map_learning_rate})
        if aux_params:
            self.actor.rnn.register_forward_hook(self.store_memory)
        self.optimizer = type(self.optimizer)(groups, lr=self.learning_rate)
        self.num_adaptive_groups = 2 if follows_actor else 1
        self.fixed_rates = [g["lr"] for g in self.optimizer.param_groups[self.num_adaptive_groups:]]

    def apply_adaptive_rate(self) -> None:
        for group in self.optimizer.param_groups[:self.num_adaptive_groups]:
            group["lr"] = self.learning_rate

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        saved = loaded_dict.get("optimizer_state_dict")
        groups = self.optimizer.param_groups
        if saved and len(saved["param_groups"]) != len(groups):
            # 旧检查点参数编号同为 actor 在前、critic 在后，仅分组不同，按当前分组重新切分
            ids = [i for group in saved["param_groups"] for i in group["params"]]
            if len(ids) != sum(len(group["params"]) for group in groups):
                raise ValueError("checkpoint optimizer parameter count does not match the model")
            regrouped, start = [], 0
            for group in groups:
                regrouped.append({**saved["param_groups"][0], "params": ids[start:start + len(group["params"])]})
                start += len(group["params"])
            loaded_dict = {**loaded_dict, "optimizer_state_dict": {**saved, "param_groups": regrouped}}
        loaded = super().load(loaded_dict, load_cfg, strict)
        self.apply_adaptive_rate()
        for group, rate in zip(self.optimizer.param_groups[self.num_adaptive_groups:], self.fixed_rates, strict=True):
            group["lr"] = rate
        return loaded

    def bound_recurrent_norms(self) -> None:
        # 门块谱范数大时门控成开关，参数微小变化即翻转后续整段记忆，KL 自适应只能把学习率压到下限
        gru = self.actor.rnn.rnn
        with torch.no_grad():
            for k in range(gru.num_layers):
                weight = getattr(gru, f"weight_hh_l{k}")
                blocks = weight.view(3, gru.hidden_size, gru.hidden_size)
                if bool(torch.linalg.matrix_norm(blocks, 2).max() > self.recurrent_norm_max):
                    u, s, vh = torch.linalg.svd(blocks)
                    blocks.copy_(u @ torch.diag_embed(s.clamp(max=self.recurrent_norm_max)) @ vh)
                if self.is_multi_gpu:
                    torch.distributed.broadcast(weight, src=0)

    def store_memory(self, module, inputs, output) -> None:
        del module, inputs
        self.memory = output

    def phase_loss(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        labels = unpad_trajectories(batch.observations["phase"], batch.masks).flatten()
        logits = self._raw_critic.phase_head(self.memory).flatten(0, -2)
        accuracy = (logits.argmax(-1) == labels).float().mean()
        return nn.functional.cross_entropy(logits, labels), accuracy

    def act(self, obs):
        actions = super().act(obs)
        if self.map_coef > 0 and self.search_map is not None:
            logits = self._raw_critic.map_head(self.memory.squeeze(0), obs[MAP_NLL])
            self.search_map.pending = ERASE_MAX * torch.sigmoid(logits)
        return actions

    def map_loss(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # 一步擦除后真实孔位格的 NLL，当前图状态来自 rollout 不回传；最优解为真实后验
        nll = batch.observations.get_non_tensor(MAP_BATCH)[MAP_NLL]
        after, before = erase_nll(self._raw_critic.map_head(self.memory, nll), nll)
        return after.mean(), (before - after).mean()

    def process_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        # 采样期间冻结归一化，否则存储的旧策略与更新时重算的不一致；整批在 update 末尾合并
        del obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def smooth_loss(self, batch, clean_mean: torch.Tensor) -> torch.Tensor:
        # 仅在悬空步扰动力旋量观测，并只在悬空输出步惩罚动作均值变化；接触时对力的响应不受约束
        obs = batch.observations
        free = obs["free_air"]
        actor_obs = obs["actor"].clone()
        # 与环境振动和传感噪声同步启用，噪声关闭阶段悬空力旋量没有需要忽略的干扰
        actor_obs[..., WRENCH_SLICE] += (torch.randn_like(actor_obs[..., WRENCH_SLICE]) * self.smooth_std
                                         * free * obs["noise_scale"])
        perturbed = obs.clone(False)
        perturbed["actor"] = actor_obs
        mean = self.actor(perturbed, masks=batch.masks, hidden_state=batch.hidden_states[0])
        weight = unpad_trajectories(free, batch.masks)
        return ((mean - clean_mean).square().sum(-1, keepdim=True) * weight).sum() / weight.sum().clamp_min(1.)

    def update(self) -> dict[str, float]:
        totals = {"value": 0., "surrogate": 0., "entropy": 0., "smooth": 0., "kl": 0.,
                  "phase": 0., "phase_accuracy": 0., "map": 0., "map_gain": 0.}
        generator = (self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
                     if self.actor.is_recurrent or self.critic.is_recurrent
                     else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))
        # 地图观测占存储大头，整体按轨迹填充会按回合数成倍复制；循环小批次按环境顺序切分，直接取环境切片
        maps = {key: self.storage.observations.pop(key) for key in MAP_KEYS
                if key in self.storage.observations.keys()}
        size = self.storage.num_envs // self.num_mini_batches
        for index, batch in enumerate(generator):
            if maps:
                envs = slice(index % self.num_mini_batches * size, (index % self.num_mini_batches + 1) * size)
                batch.observations.set_non_tensor(MAP_BATCH, {key: value[:, envs] for key, value in maps.items()})
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                       stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy
            clean_mean = self.actor.output_mean

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params,
                                                                      distribution_params))
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    totals["kl"] += kl_mean.item()
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.actor_min_lr, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    self.apply_adaptive_rate()

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantages = batch.advantages.squeeze(-1)
            surrogate_loss = torch.max(
                -advantages * ratio,
                -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2),
                                       (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            # 须在平滑正则再次前向覆盖 memory 之前计算
            phase = phase_accuracy = torch.zeros((), device=self.device)
            if self.phase_coef > 0:
                phase, phase_accuracy = self.phase_loss(batch)
                loss = loss + self.phase_coef * phase
            map_nll = map_gain = torch.zeros((), device=self.device)
            if self.map_coef > 0:
                map_nll, map_gain = self.map_loss(batch)
                loss = loss + self.map_coef * map_nll
            smooth = torch.zeros((), device=self.device)
            if self.smooth_coef > 0:
                smooth = self.smooth_loss(batch, clean_mean)
                loss = loss + self.smooth_coef * smooth

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.recurrent_norm_max is not None:
                self.bound_recurrent_norms()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["smooth"] += smooth.item()
            totals["phase"] += phase.item()
            totals["phase_accuracy"] += phase_accuracy.item()
            totals["map"] += map_nll.item()
            totals["map_gain"] += map_gain.item()
        self.storage.observations.update(maps)
        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        updates = self.num_learning_epochs * self.num_mini_batches
        return {key: value / updates for key, value in totals.items()}
