import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import entmax15

class AttentionMIL(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1, tau: float = 1.0, learnable_tau: bool = False, tau_min: float = 0.2, tau_max: float = 5.0, activation: str = 'softmax', time_aware: dict | None = None):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.learnable_tau = bool(learnable_tau)
        if self.learnable_tau:
            # store log_tau to ensure positivity; clamp at runtime to [tau_min, tau_max]
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        else:
            self.register_buffer('tau_buf', torch.tensor(float(tau)))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.activation = (activation or 'softmax').lower()

        # --- Time-aware MIL pooling options ---
        ta = time_aware or {}
        self.ta_enable: bool = bool(ta.get('enable', False))
        self.ta_mode: str = str(ta.get('mode', 'relbias')).lower()  # 'relbias' | 'concat'
        self.ta_max_patches: int = int(ta.get('max_patches', 256))
        self.ta_d_pos: int = int(ta.get('d_pos', 16))

        # Default attention scorer (no time info)
        self.v = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1)

        # Time-aware modules (created only if enabled)
        if self.ta_enable:
            if self.ta_mode == 'relbias':
                # position-dependent bias added to attention score
                self.ta_bias = nn.Embedding(self.ta_max_patches, 1)
            elif self.ta_mode == 'concat':
                # concatenate positional embedding to H for attention scoring only
                self.ta_pos = nn.Embedding(self.ta_max_patches, self.ta_d_pos)
                self.v_attn = nn.Linear(d_model + self.ta_d_pos, d_model)
                self.w_attn = nn.Linear(d_model, 1)
            else:
                # fallback to disabled if mode is unknown
                self.ta_enable = False

        # Classification head (maps pooled feature to class logits)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, H):  # H: [B,P,D]
        B, P, D = H.size()
        if self.ta_enable:
            if self.ta_mode == 'relbias':
                score = self.w(torch.tanh(self.v(H)))  # [B,P,1]
                idx = torch.arange(P, device=H.device)
                bias = self.ta_bias(idx).unsqueeze(0)  # [1,P,1]
                score = score + bias                   # [B,P,1]
            elif self.ta_mode == 'concat':
                idx = torch.arange(P, device=H.device)
                pos = self.ta_pos(idx).unsqueeze(0).expand(B, P, -1)   # [B,P,d_pos]
                Hcat = torch.cat([H, pos], dim=-1)                     # [B,P,D+d_pos]
                score = self.w_attn(torch.tanh(self.v_attn(Hcat)))     # [B,P,1]
            else:
                score = self.w(torch.tanh(self.v(H)))                  # [B,P,1]
        else:
            score = self.w(torch.tanh(self.v(H)))                      # [B,P,1]

        if self.activation == 'entmax15':
            # temperature is ignored for entmax
            A = entmax15(score, dim=1)
        else:
            if self.learnable_tau:
                tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
            else:
                tau = self.tau_buf
            A = torch.softmax(score / tau, dim=1)               # [B,P,1]
        M = (A * H).sum(dim=1)                                   # [B,D]
        logits = self.cls(self.drop(M))
        return logits, A


# --- Top-k MIL and builder ---
class TopKMIL(nn.Module):
    def __init__(self, d_model: int, n_classes: int, k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.cls = nn.Linear(d_model, n_classes)
        self.drop = nn.Dropout(dropout)
        self.k = k
    def forward(self, H):  # H: [B,P,D]
        # score patches and pick top-k
        s = self.score(H).squeeze(-1)  # [B,P]
        k = min(self.k, H.size(1))
        topk = torch.topk(s, k=k, dim=1).indices  # [B,k]
        Hk = H.gather(dim=1, index=topk.unsqueeze(-1).expand(-1, -1, H.size(-1)))  # [B,k,D]
        M = Hk.mean(dim=1)  # [B,D]
        logits = self.cls(self.drop(M))
        # pseudo-attention for visualization (1 for selected, 0 otherwise)
        A = torch.zeros_like(s).unsqueeze(-1)  # [B,P,1]
        A.scatter_(1, topk.unsqueeze(-1), 1.0)
        A = A / A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return logits, A


def build_mil_head(variant: str, d_model: int, n_classes: int, topk: int | None = None, tau: float | None = None, learnable_tau: bool | None = None, activation: str | None = None, time_aware: dict | None = None):
    variant = (variant or 'attention').lower()
    if variant in ['attention', 'attn']:
        return AttentionMIL(d_model, n_classes, tau=(tau or 1.0), learnable_tau=bool(learnable_tau), activation=(activation or 'softmax'), time_aware=time_aware)
    if variant in ['topk', 'top-k', 'top_k']:
        return TopKMIL(d_model, n_classes, k=topk or 5)
    # default
    return AttentionMIL(d_model, n_classes, tau=(tau or 1.0), learnable_tau=bool(learnable_tau), activation=(activation or 'softmax'), time_aware=time_aware)