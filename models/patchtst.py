from typing import Tuple
import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, P, patch_len*C]
        B, T, C = x.shape
        patches = []
        for start in range(0, max(T - self.patch_len + 1, 1), self.stride):
            end = start + self.patch_len
            if end > T:
                pad = torch.zeros(B, end - T, C, device=x.device, dtype=x.dtype)
                xpad = torch.cat([x, pad], dim=1)
                patches.append(xpad[:, start:end, :])
                break
            patches.append(x[:, start:end, :])
        P = len(patches)
        out = torch.stack(patches, dim=1).reshape(B, P, self.patch_len * C)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PatchTST(nn.Module):
    def __init__(self, d_model=256, n_heads=4, depth=4, patch_len=32, stride=16, dropout=0.1):
        super().__init__()
        self.patchify = Patchify(patch_len, stride)
        self.embed = nn.LazyLinear(d_model)  # 入力は各チャネルでまとめる場合は適宜変更
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=d_model*4, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.posenc = PositionalEncoding(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ここでは簡略化のため C をチャネル平均で潰す
        if x.dim() != 3:
            raise ValueError('expected [B,T,C]')
        P = self.patchify(x)                            # [B,P,patch_len*C]
        B, Pn, D = P.shape
        z = self.embed(P)                               # [B,P,d_model]
        z = self.posenc(z)
        z = self.encoder(z)                             # [B,P,d_model]
        return z