"""循环 PPO + 空中力旋量观测平滑正则：抑制策略对悬空时自身惯性/振动力波动的反应"""

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.utils import unpad_trajectories

WRENCH_SLICE = slice(7, 13)
SMOOTH_PPO = "insertion.ppo:SmoothPPO"

# TF32 下 rollout 逐步推理与更新时整段推理的误差经 GRU 累积，旧策略概率失真
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class SmoothPPO(PPO):
    def __init__(self, *args, smooth_coef: float = 0.0, smooth_force_std: float = 0.5,
                 smooth_moment_std: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.rnd or self.symmetry:
            raise ValueError("SmoothPPO does not support RND or symmetry extensions")
        self.smooth_coef = smooth_coef
        self.smooth_std = torch.tensor([smooth_force_std]*3 + [smooth_moment_std]*3, device=self.device)
        # KL 自适应学习率只调 actor；critic 不改变动作分布，随 KL 压低会拖慢价值学习
        self.optimizer = type(self.optimizer)(
            [{"params": list(self.actor.parameters())}, {"params": list(self.critic.parameters())}],
            lr=self.learning_rate)
        self.fixed_rates = [self.learning_rate]

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
        for group, rate in zip(self.optimizer.param_groups[1:], self.fixed_rates, strict=True):
            group["lr"] = rate
        return loaded

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
        totals = {"value": 0., "surrogate": 0., "entropy": 0., "smooth": 0., "kl": 0.}
        generator = (self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
                     if self.actor.is_recurrent or self.critic.is_recurrent
                     else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))
        for batch in generator:
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
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    self.optimizer.param_groups[0]["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            advantages = torch.squeeze(batch.advantages)
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

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["smooth"] += smooth.item()
        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        updates = self.num_learning_epochs * self.num_mini_batches
        return {key: value / updates for key, value in totals.items()}
