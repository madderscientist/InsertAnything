"""Privileged vertical rays for the local surface-height target, in metres."""

import mujoco_warp as mjwarp
from mujoco_warp._src.types import vec6
import torch
import warp as wp

from .spatial_memory import CELLS


class SurfaceRays:
    def __init__(self, num_envs: int, device: str, tool_body: int, ceiling: float) -> None:
        self.ceiling = ceiling
        self.origin = torch.zeros(num_envs, CELLS, 3, device=device)
        self.origin[..., 2] = ceiling
        direction = torch.zeros_like(self.origin)
        direction[..., 2] = -1
        self.distance = torch.empty(num_envs, CELLS, device=device)
        self.ray_args = (
            wp.from_torch(self.origin, dtype=wp.vec3), wp.from_torch(direction, dtype=wp.vec3),
            vec6(-1, -1, -1, -1, -1, -1), True,
            wp.full(CELLS, tool_body, dtype=wp.int32, device=device),
            wp.from_torch(self.distance), wp.empty((num_envs, CELLS), dtype=wp.int32, device=device),
            wp.empty((num_envs, CELLS), dtype=wp.vec3, device=device),
        )

    def sample(self, model, data, world_xy, board_z, depth):
        self.origin[..., :2].copy_(world_xy)
        mjwarp.rays(model, data, *self.ray_args)
        # 通孔没有底面命中，以本回合孔板下边缘作为深度截断。
        height = self.ceiling - self.distance - board_z
        return torch.where(self.distance >= 0, height.clamp_min(-depth), -depth)
