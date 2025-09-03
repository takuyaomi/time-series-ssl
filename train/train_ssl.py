import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.seed import set_seed
from utils.logger import get_logger
from utils.config import Config
from datasets.har import HARDS
from models.patchtst import PatchTST
from models.ppt_head import PPTHead



def train_loop(cfg: Config):
    logger = get_logger('SSL')
    set_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = HARDS(root=cfg.data_root, split='train')
    dl = DataLoader(ds, batch_size=cfg.train['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    # sanity-check one batch shape
    xb = next(iter(dl))
    xb = xb[0] if isinstance(xb, (list, tuple)) else xb
    logger.info(f"sample batch shape: {tuple(xb.shape)}  # expected [B, T, C]")

    model = PatchTST(**cfg.model).to(device)
    head = PPTHead(**{
        'weak_rate': cfg.ssl['order_shuffle_rate']['weak'],
        'strong_rate': cfg.ssl['order_shuffle_rate']['strong'],
        'channel_jitter': cfg.ssl.get('channel_jitter', 0.1),
    }).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train['optimizer']['lr'], weight_decay=cfg.train['optimizer']['weight_decay'])

    best = 1e9
    for epoch in range(cfg.train['epochs']):
        model.train()
        total = 0.0
        for batch in dl:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z = model(x)
            losses = head(z)
            loss = losses['total']
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        logger.info(f'Epoch {epoch+1}: ssl_loss={total/len(dl):.4f}')
        if total < best:
            best = total
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({'model': model.state_dict()}, 'checkpoints/ssl_ppt.ckpt')
            logger.info('Checkpoint saved: checkpoints/ssl_ppt.ckpt')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data_root', '--data-root', dest='data_root', type=str, required=True)
    args = ap.parse_args()
    cfg = Config.load(args.config, overrides={'data_root': args.data_root})
    train_loop(cfg)