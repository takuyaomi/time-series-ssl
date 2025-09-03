# datasets/har.py
import torch
from torch.utils.data import Dataset

class HARDS(Dataset):
    def __init__(self, root: str, split: str = "train", zscore: bool = True):
        # root = "Chief-AI-Engineer/time-series-ssl/HAR"
        data = torch.load(f"{root}/{split}.pt")  # 例: {"samples": [N,T,C], "labels": [N]} or {"samples": ...}の
        print(data)
        X = data["samples"].float()
        self.y = data.get("labels", None)  # SSL用途では y 無しでもOK
        if self.y is not None:
            self.y = self.y.long()

        if zscore:
            mu = X.mean(dim=(0, 1), keepdim=True)
            std = X.std(dim=(0, 1), keepdim=True) + 1e-6
            X = (X - mu) / std
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]                      # もともと [C, T]
        x = x.transpose(0, 1).contiguous()   # → [T, C] に統一
        if self.y is not None:
            return x, self.y[idx]
        return (x,)
        
if __name__ == "__main__":
    ds = HARDS(root="HAR", split="train")
    print("Dataset length:", len(ds))
    print("First sample shape:", ds[0][0].shape)
    if ds.y is not None:
        print("First label:", ds[0][1].item())