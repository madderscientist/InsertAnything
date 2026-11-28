"""h_t = a_t h_{t-1} + b_t 的 Triton 融合核：每个 (环境, 通道) 一个线程沿时间递推，与逐步 addcmul 逐位一致"""

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _forward(a_ptr, b_ptr, h0_ptr, out_ptr, steps, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    h = tl.load(h0_ptr + offsets, mask=mask, other=0.)
    for t in range(steps):
        a = tl.load(a_ptr + t * size + offsets, mask=mask, other=0.)
        h = a * h + tl.load(b_ptr + t * size + offsets, mask=mask, other=0.)
        tl.store(out_ptr + t * size + offsets, h, mask=mask)


@triton.jit
def _backward(a_ptr, out_ptr, h0_ptr, grad_ptr, grad_a_ptr, grad_b_ptr, grad_h0_ptr, steps, size,
              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    grad_h = tl.zeros((BLOCK,), tl.float32)
    for i in range(steps):
        t = steps - 1 - i
        grad_h += tl.load(grad_ptr + t * size + offsets, mask=mask, other=0.)
        tl.store(grad_b_ptr + t * size + offsets, grad_h, mask=mask)
        if t > 0:
            previous = tl.load(out_ptr + (t - 1) * size + offsets, mask=mask, other=0.)
        else:
            previous = tl.load(h0_ptr + offsets, mask=mask, other=0.)
        tl.store(grad_a_ptr + t * size + offsets, grad_h * previous, mask=mask)
        grad_h *= tl.load(a_ptr + t * size + offsets, mask=mask, other=0.)
    tl.store(grad_h0_ptr + offsets, grad_h, mask=mask)


class LinearScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        a, b, h0 = a.contiguous(), b.contiguous(), h0.contiguous()
        out = torch.empty_like(b)
        size = h0.numel()
        _forward[(triton.cdiv(size, BLOCK),)](a, b, h0, out, a.shape[0], size, BLOCK=BLOCK)
        ctx.save_for_backward(a, out, h0)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, out, h0 = ctx.saved_tensors
        grad_a, grad_b, grad_h0 = torch.empty_like(a), torch.empty_like(a), torch.empty_like(h0)
        size = h0.numel()
        _backward[(triton.cdiv(size, BLOCK),)](a, out, h0, grad.contiguous(), grad_a, grad_b, grad_h0,
                                                a.shape[0], size, BLOCK=BLOCK)
        return grad_a, grad_b, grad_h0
