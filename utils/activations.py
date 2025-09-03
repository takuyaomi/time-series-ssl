

import torch
from torch import Tensor

__all__ = ["entmax15"]

def entmax15(x: Tensor, dim: int = -1, n_iter: int = 50, eps: float = 1e-8) -> Tensor:
    """
    Entmax with alpha = 1.5 (a sparse, multi-peak alternative to softmax).

    This implementation uses a numerically stable, vectorized bisection to find
    the threshold \tau along the given dimension such that
        p_i = relu(z_i - \tau)^2,  where z = x / 2,
    and sum_i p_i = 1. The mapping is differentiable almost everywhere and
    works as a drop-in replacement for softmax when sparsity is beneficial.

    Args:
        x: input tensor.
        dim: dimension over which to apply the transform.
        n_iter: number of bisection iterations.
        eps: numerical stability epsilon.

    Returns:
        Tensor of the same shape as `x` with non-negative entries summing to 1
        along `dim`.
    """
    # Move dim to last for simpler broadcasting, then move back at the end
    if dim < 0:
        dim = x.dim() + dim
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm)

    z = x_perm / 2.0  # because (alpha - 1) = 0.5 for alpha = 1.5

    # Bounds for tau: lo <= tau <= hi, with sum(relu(z - tau)^2) decreasing in tau
    hi, _ = torch.max(z, dim=-1, keepdim=True)
    lo, _ = torch.min(z, dim=-1, keepdim=True)
    lo = lo - 1.0  # safe lower bound

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        p = torch.clamp(z - mid, min=0.0) ** 2
        s = p.sum(dim=-1, keepdim=True)
        # If sum > 1, tau is too small → move lo up; else move hi down
        cond = (s > 1.0)
        lo = torch.where(cond, mid, lo)
        hi = torch.where(cond, hi, mid)

    tau = hi
    p = torch.clamp(z - tau, min=0.0) ** 2

    # Renormalize (bisection should already be close to 1, but keep it exact)
    s = p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / s

    # Move dim back to original position
    inv_perm = list(range(x.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    return p.permute(*inv_perm)