"""Privileged maximum collision-surface height over each spatial memory cell."""

import mujoco
import torch
import warp as wp

from .spatial_memory import CELL_M


@wp.func
def _clip_z(bounds: wp.vec2, axis: wp.vec3, xy: wp.vec2, radius: float):
    # The square and box overlap on this axis when |dot(xy, axis.xy) + z*axis.z| <= radius.
    offset = xy[0] * axis[0] + xy[1] * axis[1]
    if axis[2] == 0.0:
        if wp.abs(offset) > radius:
            return wp.vec2(wp.inf, -wp.inf)
        return bounds
    z0 = (-radius - offset) / axis[2]
    z1 = (radius - offset) / axis[2]
    return wp.vec2(wp.max(bounds[0], wp.min(z0, z1)), wp.min(bounds[1], wp.max(z0, z1)))


@wp.func
def _box_cell_top(pos: wp.vec3, rotation: wp.mat33, size: wp.vec3, center: wp.vec2, half: float):
    # At height z the query is an axis-aligned box of half-size (half, half, 0).
    # All separating-axis constraints are intervals in z. Their highest common point is the label.
    xy = center - wp.vec2(pos[0], pos[1])
    if (wp.abs(xy[0]) > half + wp.dot(wp.abs(rotation[0]), size)
            or wp.abs(xy[1]) > half + wp.dot(wp.abs(rotation[1]), size)):
        return -wp.inf
    extent_z = wp.dot(wp.abs(rotation[2]), size)
    bounds = wp.vec2(-extent_z, extent_z)
    local_axes = wp.transpose(rotation)
    for i in range(3):
        axis = local_axes[i]
        radius = size[i] + half * (wp.abs(axis[0]) + wp.abs(axis[1]))
        bounds = _clip_z(bounds, axis, xy, radius)
        if bounds[0] > bounds[1]:
            return -wp.inf

    # For ordinary plate and rim boxes Z is shared; their edge-cross axes duplicate face axes.
    if rotation[2, 0] != 0.0 or rotation[2, 1] != 0.0:
        for i in range(3):
            world_axis = wp.vec3()
            world_axis[i] = 1.0
            for j in range(3):
                axis = wp.cross(world_axis, local_axes[j])
                radius = wp.dot(wp.abs(local_axes * axis), size) + half * (wp.abs(axis[0]) + wp.abs(axis[1]))
                bounds = _clip_z(bounds, axis, xy, radius)
                if bounds[0] > bounds[1]:
                    return -wp.inf
    return pos[2] + bounds[1]


@wp.kernel
def _surface_heights(geom_ids: wp.array(dtype=wp.int32), sizes: wp.array2d(dtype=wp.vec3),
                     positions: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.mat33),
                     centers: wp.array2d(dtype=wp.vec2), board_z: wp.array2d(dtype=wp.float32),
                     half: float, heights: wp.array2d(dtype=wp.float32)):
    world, cell, item = wp.tid()
    geom = geom_ids[item]
    top = _box_cell_top(positions[world, geom], rotations[world, geom],
                        sizes[world % sizes.shape[0], geom], centers[world, cell], half)
    if top != -wp.inf:
        wp.atomic_max(heights, world, cell, top - board_z[world, 0])


class SurfaceHeights:
    def __init__(self, model: mujoco.MjModel, tool_body: int, device: str) -> None:
        # All generated plate walls, rims and rough tiles are box colliders. Exclude the peg.
        ids = (model.geom_bodyid != tool_body).nonzero()[0]
        if (model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_BOX).any():
            raise ValueError("surface-height labels require box terrain colliders")
        self.geom_ids = wp.array(ids, dtype=wp.int32, device=device)

    def sample(self, model, data, world_xy: torch.Tensor, board_z: torch.Tensor, depth: float):
        heights = world_xy.new_full(world_xy.shape[:2], -depth)
        wp.launch(_surface_heights, dim=(*heights.shape, len(self.geom_ids)),
                  inputs=[self.geom_ids, model.geom_size, data.geom_xpos, data.geom_xmat,
                          wp.from_torch(world_xy.contiguous(), dtype=wp.vec2), wp.from_torch(board_z),
                          CELL_M / 2, wp.from_torch(heights)], device=self.geom_ids.device)
        return heights
