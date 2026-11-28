"""尖端局部 minGRU 空间记忆。坐标相对于观测目标，XY 索引顺序与旧概率图一致。"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

GRID = 48
CELL_M = 0.001
CHANNELS = 8  # 七个学习通道，以及一个仅表示该格被写过的标志
PATCH = 3
CELLS = PATCH * PATCH
BLOCK = 8
COARSE = GRID // BLOCK
FEATURES = 64
HEIGHT_UNIT_M = 0.01
HEIGHT_TARGET = "spatial_height_m"
REPLAY = "spatial_rollout"


def local_patch(tip_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """取真实格中心。窗口越界部分无效，每个窗口的存储下标仍然唯一。"""
    position = tip_xy / CELL_M + GRID / 2
    cell = position.floor().long()
    corner = (cell - PATCH // 2).clamp(0, GRID - PATCH)
    axis = torch.arange(PATCH, device=tip_xy.device)
    grid_xy = corner.unsqueeze(-2) + torch.cartesian_prod(axis, axis)
    relative = grid_xy.to(position.dtype) + 0.5 - position.unsqueeze(-2)
    valid = (grid_xy - cell.unsqueeze(-2)).abs().amax(-1) <= PATCH // 2
    indices = grid_xy[..., 0] * GRID + grid_xy[..., 1]
    return indices, relative, valid


def cell_centers(indices: torch.Tensor) -> torch.Tensor:
    grid_xy = torch.stack((indices // GRID, indices % GRID), -1)
    return (grid_xy.float() + 0.5 - GRID / 2) * CELL_M


def _coarse_delta(indices: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    bins = (indices // GRID // BLOCK) * COARSE + indices % GRID // BLOCK
    out = delta.new_zeros(*indices.shape[:-1], COARSE**2, CHANNELS)
    return out.scatter_add_(-2, bins.unsqueeze(-1).expand_as(delta), delta)


def _replay_cells(anchor, cells, episode, gate, candidate):
    """按 (环境, 回合, 格) 排序，同一格的历次写入做前缀仿射递推。

    每个格每步最多更新一次。保留跨步梯度，只分配被触及的格子，无需存 T 份全图。
    """
    steps, envs, _ = cells.shape
    key = (episode.unsqueeze(-1) * GRID**2 + cells).transpose(0, 1).reshape(envs, -1)
    order = key.argsort(dim=1, stable=True)
    env = torch.arange(envs, device=cells.device)[:, None]
    key = key[env, order]
    first = torch.ones_like(key, dtype=torch.bool)
    first[:, 1:] = key[:, 1:] != key[:, :-1]
    initial = anchor[env, key % GRID**2]
    initial = initial * (key < GRID**2).unsqueeze(-1)
    a = (1 - gate).transpose(0, 1).reshape(envs, -1, CHANNELS)[env, order]
    b = (gate * candidate).transpose(0, 1).reshape(envs, -1, CHANNELS)[env, order]
    b = b + a * (first.unsqueeze(-1) * initial)
    stride = 1
    while stride < steps:
        same = (key[:, stride:] == key[:, :-stride]).unsqueeze(-1)
        tail = b[:, stride:] + torch.where(same, a[:, stride:] * b[:, :-stride], 0.)
        b = torch.cat((b[:, :stride], tail), 1)
        a = torch.cat((a[:, :stride], torch.where(same, a[:, stride:] * a[:, :-stride], a[:, stride:])), 1)
        stride *= 2
    before = torch.where(first.unsqueeze(-1), initial, b.roll(1, dims=1))
    inverse = torch.empty_like(order)
    inverse.scatter_(1, order, torch.arange(order.shape[1], device=cells.device).expand_as(order))
    shape = (envs, steps, CELLS, CHANNELS)
    after = b[env, inverse].reshape(shape).transpose(0, 1)
    delta = (b - before)[env, inverse].reshape(shape).transpose(0, 1)
    return after, delta


class SpatialMemory(nn.Module):
    """每格保留有符号特征及写入标志。写门和候选值由时序记忆和格心相对尖端的偏移决定。"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.evidence = nn.Linear(latent_dim, 32)
        self.offset = nn.Linear(2, 32, bias=False)
        self.write = nn.Linear(32, 2 * (CHANNELS - 1))
        self.read = nn.Sequential(nn.Linear((CELLS + COARSE**2) * CHANNELS + 2, FEATURES), nn.ELU())
        self.height = nn.Linear(CHANNELS, 1, bias=False)
        nn.init.normal_(self.height.weight, std=0.02)
        # 运行时缓冲按环境数分配。检查点只保存网络，恢复时进入新回合。
        self.grid = None
        self.coarse = None
        self.patch_prediction = self.patch_valid = None

    @property
    def flat_grid(self):
        return self.grid.permute(0, 2, 3, 1).flatten(1, 2)

    def initialize(self, like: torch.Tensor) -> None:
        if self.grid is None or self.grid.shape[0] != like.shape[0] or self.grid.device != like.device:
            self.grid = like.new_zeros(like.shape[0], GRID, GRID, CHANNELS).permute(0, 3, 1, 2)
            self.coarse = like.new_zeros(like.shape[0], COARSE**2, CHANNELS)

    def reset(self, dones=None) -> None:
        if dones is None:
            self.grid = self.coarse = None
        elif self.grid is not None:
            ids = dones.reshape(-1).bool().nonzero().flatten()
            self.grid.index_fill_(0, ids, 0.)
            self.coarse.index_fill_(0, ids, 0.)

    def forward(self, memory, tip_xy, replay=None, *, predict_height=True):
        cells, relative, valid = local_patch(tip_xy)
        evidence = self.evidence(memory).unsqueeze(-2) + self.offset(relative.clamp(-PATCH, PATCH))
        gate, candidate = self.write(torch.nn.functional.elu(evidence)).chunk(2, dim=-1)
        written = torch.ones_like(gate[..., :1])
        gate = torch.cat((gate.sigmoid(), written), -1) * valid.unsqueeze(-1)
        candidate = torch.cat((candidate.tanh(), written), -1)
        if replay is None:
            self.initialize(tip_xy)
            env = torch.arange(tip_xy.shape[0], device=tip_xy.device)[:, None]
            flat = self.flat_grid
            before = flat[env, cells]
            after = before + gate * (candidate - before)
            delta = _coarse_delta(cells, after - before)
            with torch.no_grad():
                flat[env, cells] = after.detach()
                self.coarse.add_(delta.detach())
            coarse = self.coarse
        else:
            anchor, coarse_start, dones = replay
            # 第 t 步的 done 在第 t+1 步清空所有空间记忆。首步状态来自采样起点。
            starts = torch.cat((torch.zeros_like(dones[:1]), dones[:-1]), 0)
            episode = starts.long().cumsum(0)
            after, delta = checkpoint(_replay_cells, anchor, cells, episode, gate, candidate,
                                      use_reentrant=False, preserve_rng_state=False)
            changes = _coarse_delta(cells, delta).cumsum(0)
            # 从最近一次 reset 开始累加；同一回合中的重访保留之前写入的梯度。
            time = torch.arange(cells.shape[0], device=cells.device)[:, None]
            last = torch.where(starts, time, -1).cummax(0).values
            env = torch.arange(cells.shape[1], device=cells.device)
            prefix = changes[(last - 1).clamp_min(0), env]
            coarse = changes + torch.where((last < 0)[..., None, None], coarse_start, -prefix)
        local = after * valid.unsqueeze(-1)
        if replay is not None and predict_height:
            self.patch_prediction = self.height(after).squeeze(-1)
            self.patch_valid = valid
        else:
            self.patch_prediction = self.patch_valid = None
        features = torch.cat((local.flatten(-2), (coarse / BLOCK).flatten(-2),
                              (tip_xy / (GRID * CELL_M / 2)).clamp(-2, 2)), -1)
        return self.read(features)

    @torch.no_grad()
    def decoded_surface(self, index: int):
        """返回一个环境的预测高度（米）和已写格掩码，供回放显示。"""
        grid = self.grid[index].permute(1, 2, 0)
        return self.height(grid).squeeze(-1) * HEIGHT_UNIT_M, grid[..., -1] > 0.5

    def height_loss(self, target_m):
        # 图外没有监督。零有效格也返回可微的零损失。
        valid = self.patch_valid
        error = self.patch_prediction - target_m / HEIGHT_UNIT_M
        cost = torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=0.1)
        count = valid.sum().clamp_min(1)
        return (cost * valid).sum() / count, (error.detach().abs() * valid).sum() * (1000 * HEIGHT_UNIT_M) / count
