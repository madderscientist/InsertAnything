"""批量 wxyz 四元数旋转与参考点力旋量坐标变换，统一使用国际单位制"""

import torch
from torch import Tensor


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_inv(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def canonical_quat(q: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    q = q / norm.clamp_min(1e-12)
    # 按字典序首个非零分量统一符号，消除 q 与 -q 的双重覆盖
    sign = torch.ones_like(q[..., 0])
    undecided = torch.ones_like(sign, dtype=torch.bool)
    for i in range(4):
        nonzero = q[..., i].abs() > 1e-8
        sign = torch.where(undecided & nonzero, torch.sign(q[..., i]), sign)
        undecided = undecided & ~nonzero
    return q * sign.unsqueeze(-1)


def rotate(q: Tensor, v: Tensor) -> Tensor:
    uv = torch.linalg.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv))


def rotvec_quat(v: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(angle / (2 * torch.pi))
    return torch.cat((torch.cos(angle / 2), v * scale), -1)


def quat_rotvec(q: Tensor) -> Tensor:
    q = canonical_quat(q)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[..., :1])
    scale = torch.where(sine > 1e-7, angle / sine.clamp_min(1e-12),
                        torch.full_like(sine, 2.0))
    return q[..., 1:] * scale


def transform_wrench(force: Tensor, moment: Tensor, q_target_source: Tensor,
                     source_to_target_in_target: Tensor) -> Tensor:
    # 先旋转坐标轴，再按源点到目标点的位移附加力臂叉乘力矩
    f = rotate(q_target_source, force)
    m = rotate(q_target_source, moment)
    return torch.cat((f, m - torch.linalg.cross(source_to_target_in_target, f)), -1)


def tip_pose(body_pos: Tensor, q: Tensor, body_to_tip: Tensor) -> Tensor:
    return body_pos + rotate(q, body_to_tip)


def action_target(tip_pos: Tensor, q: Tensor, action: Tensor,
                  translation_limit: float, rotation_limit: float) -> tuple[Tensor, Tensor]:
    # 在孔坐标系下叠加增量，纯旋转时保持插头尖端位置不变
    a = action.clamp(-1, 1)
    p_target = tip_pos + a[..., :3] * translation_limit
    q_target = canonical_quat(quat_mul(rotvec_quat(a[..., 3:] * rotation_limit), q))
    return p_target, q_target


def actor_observation(tip_pos: Tensor, q: Tensor, target_pos: Tensor,
                      target_q: Tensor, wrench: Tensor) -> Tensor:
    return torch.cat((target_pos - tip_pos,
                      canonical_quat(quat_mul(quat_inv(q), target_q)), wrench), -1)
