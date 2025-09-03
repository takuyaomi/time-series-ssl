import torch
import torch.nn as nn
import torch.nn.functional as F


def _permute_order(z: torch.Tensor, rate: float, channel_jitter: float = 0.0) -> torch.Tensor:
    # z: [B,P,D]
    B, P, D = z.shape
    k = int(P * rate)
    idx = torch.arange(P, device=z.device).unsqueeze(0).repeat(B,1)
    if k > 0:
        perm = torch.stack([torch.randperm(P, device=z.device) for _ in range(B)], dim=0)
        mask = torch.zeros_like(idx, dtype=torch.bool)
        mask[:, :k] = True
        mask = mask[torch.arange(B).unsqueeze(1), torch.randperm(P, device=z.device).unsqueeze(0).repeat(B,1)]
        idx = torch.where(mask, perm, idx)
    z_perm = z[torch.arange(B).unsqueeze(1), idx]
    if channel_jitter > 0:
        z_perm = z_perm + channel_jitter * torch.randn_like(z_perm)
    return z_perm


def info_nce(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    q = F.normalize(q.mean(dim=1), dim=-1)  # [B,D]
    k = F.normalize(k.mean(dim=1), dim=-1)
    logits = q @ k.t() / temperature        # [B,B]
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def order_consistency_loss(z: torch.Tensor, zw: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
    # 近似: 弱より強の方が順序破壊が大きいという順位関係を学習
    def pair_sim(a, b):
        return F.cosine_similarity(a[:,1:], b[:,:-1], dim=-1).mean()
    s_orig = pair_sim(z, z)
    s_w = pair_sim(z, zw)
    s_s = pair_sim(z, zs)
    return F.relu((s_w - s_orig)) + F.relu((s_s - s_w))


class PPTHead(nn.Module):
    def __init__(self, weak_rate=0.2, strong_rate=0.6, channel_jitter=0.1):
        super().__init__()
        self.weak_rate = weak_rate
        self.strong_rate = strong_rate
        self.channel_jitter = channel_jitter

    def forward(self, z):  # z: [B,P,D]
        z_weak = _permute_order(z, self.weak_rate, 0.0)
        z_strong = _permute_order(z, self.strong_rate, self.channel_jitter)
        l_contrast = info_nce(z_weak, z_strong)
        l_order = order_consistency_loss(z, z_weak, z_strong)
        return {"contrast": l_contrast, "order": l_order, "total": l_contrast + l_order}