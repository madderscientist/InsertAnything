"""空间概率记忆 C1：按目标偏差初始化的孔位概率图，只允许学习到的局部擦除（减法）再归一化

地图位于不准确目标点坐标系（观测尖端坐标），每格 1 mm；对数域累积擦除量 s ≤ 0，
P = softmax(log P0 + s)。每步在尖端所在格附近 5×5 区域按擦除概率 e 更新
s ← max(s + log(1-e), -L)，e 由 actor GRU 潜变量经擦除头给出，以一步 NLL 训练
"""

import math

import torch
import torch.nn as nn

MAP_FINE, MAP_COARSE, MAP_NLL = "map_fine", "map_coarse", "map_nll"
# PPO 更新时未填充的地图观测，以非张量条目随批次传入，布局同 unpad_trajectories 的输出
MAP_BATCH = "map_batch"
MAP_KEYS = (MAP_FINE, MAP_COARSE, MAP_NLL)
CELL_M = 0.001
GRID = 48
PATCH = 5
PATCH_CELLS = PATCH * PATCH
FINE = 9
POOL = 4
COARSE = GRID // POOL
ERASE_MAX = 0.5
LOG_FLOOR = math.log(100.)
MAP_FEATURES = 64
DENSITY_MIN, DENSITY_MAX, DENSITY_SCALE = -8., 6., 4.


def prior_log_mass(std: torch.Tensor) -> torch.Tensor:
    """各向同性高斯先验的格质量，std 为目标偏差与夹持、标定误差合成的每轴标准差"""
    edges = (torch.arange(GRID + 1, device=std.device) - GRID / 2) * CELL_M
    cdf = torch.special.ndtr(edges[None, :] / std.clamp_min(CELL_M / 2)[:, None])
    axis = cdf[:, 1:] - cdf[:, :-1]
    mass = axis[:, :, None] * axis[:, None, :]
    return torch.log_softmax(mass.clamp_min(1e-9).log().flatten(1), -1).view_as(mass)


def erase_nll(logits: torch.Tensor, nll: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """擦除后真实格的负对数概率，以及擦除前的值；P_t(c*) = P(c*) f(c*) / (1 - Σ P_i (1 - f_i))"""
    p = nll[..., :PATCH_CELLS]
    erased = nll[..., PATCH_CELLS:2*PATCH_CELLS]
    hit = nll[..., 2*PATCH_CELLS:3*PATCH_CELLS]
    log_true = nll[..., 3*PATCH_CELLS]
    log_keep = torch.log1p(-ERASE_MAX * torch.sigmoid(logits))
    log_f = torch.maximum(log_keep, -(erased + LOG_FLOOR))
    removed = (p * -torch.expm1(log_f)).sum(-1)
    after = log_true + (hit * log_f).sum(-1) - torch.log1p(-removed)
    return -after, -log_true


class MapHead(nn.Module):
    """擦除头：actor GRU 潜变量 + 局部已擦除量 + 尖端相对局部区域中心的格偏移 → 5×5 擦除 logits"""

    def __init__(self, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + PATCH_CELLS + 2, hidden), nn.ELU(),
                                 nn.Linear(hidden, PATCH_CELLS))
        # 初始擦除概率约 1%，地图从先验起步
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.)

    def forward(self, latent: torch.Tensor, nll: torch.Tensor) -> torch.Tensor:
        erased = nll[..., PATCH_CELLS:2*PATCH_CELLS] / LOG_FLOOR
        offset = nll[..., -2:]
        return self.net(torch.cat((latent, erased, offset), -1))


class MapEncoder(nn.Module):
    """局部 9×9 细图（密度、访问）与全局 12×12 粗图（密度、访问、尖端）各经小 CNN 后融合"""

    def __init__(self) -> None:
        super().__init__()
        self.fine = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ELU(inplace=True),
                                  nn.Conv2d(16, 32, 3, stride=2), nn.ELU(inplace=True), nn.Flatten())
        self.coarse = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ELU(inplace=True),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(inplace=True),
                                    nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(inplace=True),
                                    nn.Flatten())
        fine_dim = 32 * ((FINE - 3) // 2 + 1) ** 2
        coarse_dim = 32 * math.ceil(COARSE / 4) ** 2
        self.out = nn.Sequential(nn.Linear(fine_dim + coarse_dim, MAP_FEATURES), nn.ELU())

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        lead = fine.shape[:-3]
        # 小通道卷积的反向受访存限制，GPU 上用 bf16 与 NHWC 布局
        with torch.autocast(fine.device.type, torch.bfloat16, enabled=fine.is_cuda):
            fine = self.fine(fine.reshape(-1, *fine.shape[-3:]).float()
                             .contiguous(memory_format=torch.channels_last))
            coarse = self.coarse(coarse.reshape(-1, *coarse.shape[-3:]).float()
                                 .contiguous(memory_format=torch.channels_last))
            features = self.out(torch.cat((fine, coarse), -1))
        return features.float().view(*lead, MAP_FEATURES)


class SearchMap:
    """逐环境概率图状态；observe 为纯观测（仅缓存局部区域索引），apply 执行待定擦除"""

    def __init__(self, num_envs: int, device: str) -> None:
        shape = (num_envs, GRID, GRID)
        self.log_prior = torch.zeros(shape, device=device)
        self.erased = torch.zeros(shape, device=device)
        self.visits = torch.zeros(shape, device=device)
        # 擦除已排除的先验质量 Σ P0 (1 - e^s)，作覆盖势函数
        self.removed = torch.zeros(num_envs, device=device)
        self.env_ids = torch.arange(num_envs, device=device)
        self.offsets = torch.arange(PATCH, device=device)
        self.fine_offsets = torch.arange(FINE, device=device)
        self.cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # 尖端在图内的连续格坐标，仅供可视化
        self.position: torch.Tensor | None = None
        self.pending: torch.Tensor | None = None
        self.metrics: dict[str, torch.Tensor] = {}

    def reset(self, ids: torch.Tensor, log_prior: torch.Tensor) -> None:
        self.log_prior[ids] = log_prior
        self.erased.index_fill_(0, ids, 0.)
        self.visits.index_fill_(0, ids, 0.)
        self.removed.index_fill_(0, ids, 0.)

    def observe(self, tip: torch.Tensor, truth: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.env_ids
        logits = self.log_prior + self.erased
        lse = torch.logsumexp(logits.flatten(1), -1)
        p = torch.exp(logits - lse[:, None, None])
        position = tip / CELL_M + GRID / 2
        cell = position.floor().long().clamp(0, GRID - 1)
        corner = (cell - PATCH // 2).clamp(0, GRID - PATCH)
        offset = (position - corner - PATCH / 2).clamp(-PATCH, PATCH)
        rows = corner[:, :1] + self.offsets
        cols = corner[:, 1:] + self.offsets
        patch = (n[:, None, None], rows[:, :, None], cols[:, None, :])
        target = (truth / CELL_M + GRID / 2).floor().long().clamp(0, GRID - 1)
        hit = (rows == target[:, :1])[:, :, None] & (cols == target[:, 1:])[:, None, :]
        log_true = logits[n, target[:, 0], target[:, 1]] - lse
        self.cache = (rows, cols, cell)
        self.position = position
        gain = log_true - self.log_prior[n, target[:, 0], target[:, 1]]
        self.metrics = {"map/info_gain": gain.mean(),
                        "map/false_erase": (gain < math.log(0.1)).float().mean(),
                        "map/entropy": (lse - (p * logits).sum((1, 2))).mean()}
        nll = torch.cat((p[patch].flatten(1), self.erased[patch].flatten(1), hit.float().flatten(1),
                         log_true[:, None], offset), -1)
        # 只在局部窗口上求密度，图外按零质量、零访问填充
        fine_rows = cell[:, :1] - FINE // 2 + self.fine_offsets
        fine_cols = cell[:, 1:] - FINE // 2 + self.fine_offsets
        inside = (((fine_rows >= 0) & (fine_rows < GRID))[:, :, None]
                  & ((fine_cols >= 0) & (fine_cols < GRID))[:, None, :])
        window = (n[:, None, None], fine_rows.clamp(0, GRID - 1)[:, :, None], fine_cols.clamp(0, GRID - 1)[:, None, :])
        density = ((logits[window] - lse[:, None, None] + math.log(GRID * GRID))
                   .clamp(DENSITY_MIN, DENSITY_MAX) / DENSITY_SCALE)
        fine = torch.stack((torch.where(inside, density, DENSITY_MIN / DENSITY_SCALE),
                            torch.where(inside, torch.log1p(self.visits[window]) / DENSITY_SCALE, 0.)), 1)
        block = p.view(-1, COARSE, POOL, COARSE, POOL).sum((2, 4))
        coarse_density = ((block * COARSE * COARSE).clamp_min(1e-12).log()
                          .clamp(DENSITY_MIN, DENSITY_MAX) / DENSITY_SCALE)
        coarse_visits = torch.log1p(self.visits.view(-1, COARSE, POOL, COARSE, POOL).sum((2, 4))) / DENSITY_SCALE
        tip_plane = torch.zeros_like(coarse_visits)
        coarse_cell = cell // POOL
        tip_plane[n, coarse_cell[:, 0], coarse_cell[:, 1]] = 1.
        coarse = torch.stack((coarse_density, coarse_visits, tip_plane), 1)
        return {MAP_FINE: fine.half(), MAP_COARSE: coarse.half(), MAP_NLL: nll}

    def apply(self) -> None:
        if self.cache is None:
            return
        rows, cols, cell = self.cache
        n = self.env_ids
        if self.pending is not None:
            patch = (n[:, None, None], rows[:, :, None], cols[:, None, :])
            keep = torch.log1p(-self.pending.view(-1, PATCH, PATCH))
            old = self.erased[patch]
            new = (old + keep).clamp_min(-LOG_FLOOR)
            self.erased[patch] = new
            self.removed += (self.log_prior[patch].exp() * (old.exp() - new.exp())).sum((1, 2))
            self.pending = None
        self.visits[n, cell[:, 0], cell[:, 1]] += 1.
