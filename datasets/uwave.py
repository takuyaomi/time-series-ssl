from typing import Tuple, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UWaveDataset(Dataset):
    """
    Gesture(UWave) dataset loader for .pt files stored as:
      root/
        train.pt
        val.pt
        test.pt
    Each .pt is expected to be a dict containing at least samples (X) and optionally labels (y).
    Accepted key aliases:
      X:  'X', 'samples', 'data'
      y:  'y', 'labels', 'target', 'targets'
    Shapes accepted for X per item: [T, C] or [C, T] (internally unified to [T, C]).
    """
    def __init__(self, root: str, split: str = 'train',
                 window_length: Optional[int] = None, hop_length: Optional[int] = None,
                 zscore: bool = False, augment: bool = False, aug_params: Optional[dict] = None):
        assert split in {'train', 'test', 'val', 'valid', 'validation'}
        fname = 'val.pt' if split in {'val', 'valid', 'validation'} else f'{split}.pt'
        fpath = os.path.join(root, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"UWaveDataset: file not found: {fpath}")
        data = torch.load(fpath, map_location='cpu')
        # resolve keys
        def get_first(d, keys, required=True):
            for k in keys:
                if k in d:
                    return d[k]
            if required:
                raise KeyError(f"Expected one of keys {keys} in {list(d.keys())}")
            return None
        X = get_first(data, ['X', 'samples', 'data']).float()  # [N, T, C] or [N, C, T]
        y = get_first(data, ['y', 'labels', 'target', 'targets'], required=False)
        if y is not None:
            y = y.long()
            y = y.view(-1)
        # unify to [N, T, C]
        if X.dim() != 3:
            raise ValueError(f"Expected X to be 3D [N,T,C] or [N,C,T], got {X.shape}")
        N, A, B = X.shape
        # Heuristic: if middle dim = channels small (<=8) and last is long (>=32), assume [N,C,T] and transpose
        if A <= 8 and B >= 16:
            X = X.transpose(1, 2).contiguous()  # [N,T,C]
        # z-score per channel using train stats only (here per split for simplicity)
        if zscore:
            mu = X.mean(dim=(0, 1), keepdim=True)
            std = X.std(dim=(0, 1), keepdim=True) + 1e-6
            X = (X - mu) / std
        self.X = X.numpy().astype(np.float32)  # keep numpy for window ops
        self.y = y.numpy().astype(np.int64) if y is not None else None
        self.window_length = window_length
        self.hop_length = hop_length
        self.split = split
        self.augment = augment
        self.aug_params = aug_params or {}

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()  # [T, C]
        # ensure [T, C]
        if x.shape[0] < x.shape[1]:
            # if somehow [C, T]
            x = x.T
        # optional fixed window (MIL-friendly). For simple baseline, take first window or pad.
        if self.window_length is not None and self.hop_length is not None:
            T = x.shape[0]
            if T < self.window_length:
                pad = np.zeros((self.window_length - T, x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
            x = x[:self.window_length]
        # light augmentations (train split only)
        if self.split in {'train'} and self.augment:
            jp = float(self.aug_params.get('jitter_std', 0.0))
            sp = float(self.aug_params.get('scale_std', 0.0))
            sh = int(self.aug_params.get('time_shift', 0))
            # jitter (Gaussian noise)
            if jp > 0:
                x = x + np.random.normal(0.0, jp, size=x.shape).astype(np.float32)
            # scaling per-channel
            if sp > 0:
                scale = np.random.normal(1.0, sp, size=(1, x.shape[1])).astype(np.float32)
                x = x * scale
            # circular time shift
            if sh > 0:
                s = np.random.randint(-sh, sh + 1)
                if s != 0:
                    x = np.roll(x, shift=s, axis=0)
        xt = torch.from_numpy(x)
        if self.y is None:
            return xt, torch.tensor(-1, dtype=torch.long)
        return xt, torch.tensor(self.y[idx], dtype=torch.long)

if __name__ == '__main__':
    # Quick smoke test: adjust the path below if you want to run this file directly
    root = os.path.join(os.path.dirname(__file__), '..', 'data', 'uwave')
    try:
        ds = UWaveDataset(root="Gesture", split='train', augment=True, aug_params={'jitter_std':0.01,'time_shift':5,'scale_std':0.05})
        print('Dataset length:', len(ds))
        x0, y0 = ds[0]
        print('First sample shape:', x0.shape)  # expect [T, C]
        print('First label:', int(y0))
    except Exception as e:
        print('UWaveDataset init error:', e)