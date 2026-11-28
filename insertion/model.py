"""minGRU 记忆 + 当前观测跳连的循环模型：记忆输出与归一化当前观测拼接再进 MLP"""

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from rsl_rl.models import RNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from rsl_rl.utils import unpad_trajectories
from torch.distributions import Normal
from torch.utils.checkpoint import checkpoint

from .search_map import MAP_BATCH, MAP_COARSE, MAP_FEATURES, MAP_FINE, MapEncoder

SKIP_RNN_MODEL = "insertion.model:SkipRNNModel"
GSDE_DISTRIBUTION = "insertion.model:GSDEDistribution"
# 50 Hz 控制频率下约 10 s
CHRONO_MAX_STEPS = 500.


class GSDEDistribution(GaussianDistribution):
    """广义状态相关探索 gSDE：噪声 = 单位化末层特征 × 每环境定期重抽的探索矩阵

    给定状态时动作服从 N(μ, σ̂²)，σ̂_j² = Σ_i z_i² σ_ij²；PPO 的似然、熵与 KL 均按此对角高斯计算
    """

    def __init__(self, output_dim: int, init_std: float = 0.4,
                 std_range: tuple[float, float] = (0.02, 1.0), resample_steps: int = 25) -> None:
        Distribution.__init__(self, output_dim)
        if resample_steps < 1:
            raise ValueError("resample_steps must be positive")
        self.init_std = init_std
        self.std_range = (max(float(std_range[0]), 1e-6), float(std_range[1]))
        self.resample_steps = resample_steps
        self.register_parameter("log_std", None)
        self.features: torch.Tensor | None = None
        self.raw_std: torch.Tensor | None = None
        self.epsilon: torch.Tensor | None = None
        self.steps = 0
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        # 特征单位化后 Σ z_i² = 1，矩阵均匀初始化即得 σ̂ = init_std
        self.log_std = nn.Parameter(torch.full((mlp[-1].in_features, self.output_dim), math.log(self.init_std)))

    def update_features(self, mean: torch.Tensor, features: torch.Tensor) -> None:
        z = features.detach()
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-6)
        self.features = z
        self.raw_std = (z.square() @ (2 * self.log_std).exp()).sqrt()
        self._distribution = Normal(mean, self.raw_std.clamp(*self.std_range))

    def sample(self) -> torch.Tensor:
        z, mean = self.features, self._distribution.mean
        if self.epsilon is None or self.epsilon.shape[0] != z.shape[0] or self.steps % self.resample_steps == 0:
            self.epsilon = torch.randn(z.shape[0], *self.log_std.shape, device=z.device)
        self.steps += 1
        noise = torch.einsum("nd,nda->na", z, self.epsilon * self.log_std.exp())
        # σ̂ 被截断时同比缩放噪声，使采样与似然一致
        return mean + noise * (self._distribution.stddev / self.raw_std.clamp_min(1e-12))

    def resample(self, dones: torch.Tensor | None = None) -> None:
        if self.epsilon is None or dones is None:
            self.epsilon = None
            return
        fresh = torch.randn_like(self.epsilon)
        self.epsilon = torch.where(dones.bool().view(-1, 1, 1), fresh, self.epsilon)


class MinGRU(nn.Module):
    """多层 minGRU：z = σ(W_z x)，h̃ = tanh(W_h x)，h = (1-z) h + z h̃，接口同 nn.GRU（时间优先）

    门与候选只看本层输入，状态 Jacobian 为 diag(1-z)，范数 < 1，参数微小变化不会经门反馈沿序列放大
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, max_steps: float) -> None:
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.layers = nn.ModuleList(nn.Linear(input_size if k == 0 else hidden_size, 2 * hidden_size)
                                    for k in range(num_layers))
        with torch.no_grad():
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[:hidden_size])
                nn.init.xavier_uniform_(layer.weight[hidden_size:])
                layer.bias.zero_()
                # chrono 初始化：保留比例 1-z = u/(1+u)，u ~ U(1, T-1)；默认初始化 z≈0.5 记忆只有约 2 步
                layer.bias[:hidden_size] = -torch.empty(hidden_size).uniform_(1, max_steps - 1).log()

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if h0 is None:
            h0 = x.new_zeros(self.num_layers, x.shape[1], self.hidden_size)
        last: List[torch.Tensor] = []
        for k, layer in enumerate(self.layers):
            y = layer(x)
            gate = y[..., :self.hidden_size]
            retain = torch.sigmoid(-gate)
            write = torch.sigmoid(gate) * torch.tanh(y[..., self.hidden_size:])
            if x.is_cuda and x.shape[0] > 1 and not torch.jit.is_scripting():
                x = self.scan(retain, write, h0[k])
            else:
                h = h0[k]
                out: List[torch.Tensor] = []
                for t in range(x.shape[0]):
                    h = torch.addcmul(write[t], retain[t], h)
                    out.append(h)
                x = torch.stack(out)
            last.append(x[-1])
        return x, torch.stack(last)

    @torch.jit.unused
    def scan(self, retain: torch.Tensor, write: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        # 逐步 Python 循环的上百次小核启动使每轮学习慢一倍；融合核与 rollout 逐步递推逐位一致
        from insertion.linear_scan import LinearScan
        return LinearScan.apply(retain, write, h0)


class SkipRNNModel(RNNModel):
    def __init__(self, *args, search_map: bool = False, **kwargs) -> None:
        # 概率图编码与 minGRU 并联接入 MLP，维度须在父类构建 MLP 前确定
        self.map_dim = MAP_FEATURES if search_map else 0
        super().__init__(*args, **kwargs)
        # 只借用 rsl_rl RNN 封装的隐状态管理；GRU 的门依赖隐状态，训练中递归权重增长后策略对参数极度敏感
        rnn = self.rnn.rnn
        self.rnn.rnn = MinGRU(rnn.input_size, rnn.hidden_size, rnn.num_layers, CHRONO_MAX_STEPS)
        if self.map_dim:
            self.map_encoder = MapEncoder()
        self.map_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, obs, masks: torch.Tensor | None = None, hidden_state=None,
                stochastic_output: bool = False) -> torch.Tensor:
        if not isinstance(self.distribution, GSDEDistribution):
            return super().forward(obs, masks, hidden_state, stochastic_output)
        *body, head = self.mlp
        features = self.get_latent(obs, masks, hidden_state)
        for layer in body:
            features = layer(features)
        mean = head(features)
        if not stochastic_output:
            return mean
        self.distribution.update_features(mean, features)
        # 训练批次（带 masks）只需分布参数，探索矩阵按环境保存只用于逐步 rollout
        return mean if masks is not None else self.distribution.sample()

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        super().reset(dones, hidden_state)
        if isinstance(self.distribution, GSDEDistribution) and hidden_state is None:
            self.distribution.resample(dones)

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        current = MLPModel.get_latent(self, obs)
        memory = self.rnn(current, masks, hidden_state).squeeze(0)
        if masks is not None:
            current = unpad_trajectories(current, masks)
        if not self.map_dim:
            return torch.cat((memory, current), -1)
        if masks is None:
            return torch.cat((memory, current, self.map_encoder(obs[MAP_FINE], obs[MAP_COARSE])), -1)
        maps = obs.get_non_tensor(MAP_BATCH)
        # 平滑正则只扰动力旋量，同一批次的地图特征直接复用；反向时重算卷积不保存激活
        if self.map_cache is None or self.map_cache[0] is not maps[MAP_FINE]:
            self.map_cache = (maps[MAP_FINE], checkpoint(self.map_encoder, maps[MAP_FINE], maps[MAP_COARSE],
                                                         use_reentrant=False))
        return torch.cat((memory, current, self.map_cache[1]), -1)

    def _get_latent_dim(self) -> int:
        return self.latent_dim + self.obs_dim + self.map_dim

    def as_jit(self) -> nn.Module:
        if self.map_dim:
            raise NotImplementedError("search map policies are not exportable yet")
        return _JitSkipRNN(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        if self.map_dim:
            raise NotImplementedError("search map policies are not exportable yet")
        return _OnnxSkipRNN(self)


class _ExportSkipRNN(nn.Module):
    def __init__(self, model: SkipRNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (model.distribution.as_deterministic_output_module()
                                     if model.distribution is not None else nn.Identity())
        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def step(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_normalizer(obs)
        memory, hidden = self.rnn(x.unsqueeze(0), hidden)
        latent = torch.cat((memory.squeeze(0), x), -1)
        return self.deterministic_output(self.mlp(latent)), hidden


class _JitSkipRNN(_ExportSkipRNN):
    def __init__(self, model: SkipRNNModel) -> None:
        super().__init__(model)
        self.register_buffer("hidden_state", torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hidden = self.step(x, self.hidden_state)
        self.hidden_state[:] = hidden
        return out

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0


class _OnnxSkipRNN(_ExportSkipRNN):
    is_recurrent: bool = True

    def forward(self, obs: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(obs, h_in)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, self.input_size), torch.zeros(self.num_layers, 1, self.hidden_size)

    @property
    def input_names(self) -> list[str]:
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]
