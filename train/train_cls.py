import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.seed import set_seed
from utils.logger import get_logger
from utils.config import Config
from datasets.uwave import UWaveDataset
from models.patchtst import PatchTST
from models.mil_head import AttentionMIL, build_mil_head
from eval.metrics import compute_metrics
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_


def train_eval(cfg: Config):
    logger = get_logger('CLS')
    set_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use full sequence; PatchTST will patchify internally
    aug_cfg = cfg.train.get('aug', {}) if isinstance(cfg.train, dict) else {}
    train_ds = UWaveDataset(root=cfg.data_root, split='train', augment=bool(aug_cfg), aug_params=aug_cfg)
    val_ds   = UWaveDataset(root=cfg.data_root, split='val')
    test_ds  = UWaveDataset(root=cfg.data_root, split='test')

    nw = cfg.train.get('num_workers', 0)
    bs = cfg.train['batch_size']

    # Optional class-balanced sampler for training
    sampler = None
    if bool(cfg.train.get('balance', False)) and hasattr(train_ds, 'y') and train_ds.y is not None:
        y_np = np.asarray(train_ds.y)
        classes_sampler, counts = np.unique(y_np, return_counts=True)
        class_weights_sampler = counts.sum() / (len(classes_sampler) * counts)
        sample_weights = class_weights_sampler[y_np].astype(np.float32)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(y_np),
            replacement=True,
        )
        logger.info(f'enable WeightedRandomSampler (class counts={counts.tolist()})')

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=(sampler is None), num_workers=nw, sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw)

    xb, yb = next(iter(train_dl)); logger.info(f"train batch shape: {tuple(xb.shape)}  # expected [B, T, C]")
    xb, yb = next(iter(val_dl));   logger.info(f"val   batch shape: {tuple(xb.shape)}  # expected [B, T, C]")

    # class weights for imbalance (Macro-F1 improvement)
    os.makedirs('reports', exist_ok=True)
    classes = np.unique(train_ds.y)
    cls_weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=train_ds.y)
    class_weights = torch.tensor(cls_weights_np, dtype=torch.float, device=device)

    logger.info(f'class weights: {cls_weights_np.tolist()}')

    model = PatchTST(**cfg.model).to(device)
    if cfg.train.get('pretrained'):  # ssl ckpt
        sd = torch.load(cfg.train['pretrained'], map_location='cpu')['model']
        model.load_state_dict(sd, strict=False)
        logger.info(f'Loaded SSL weights from {cfg.train["pretrained"]}')

    n_classes = int(max(train_ds.y.max(), test_ds.y.max()) + 1)

    mil_variant = cfg.train.get('mil_variant', 'attention')
    topk_k = cfg.train.get('topk_k', 5)

    # Read temperature from either legacy key `mil_tau` or nested `train.mil.temperature`
    mil_tau_legacy = cfg.train.get('mil_tau', None)
    mil_cfg_train = cfg.train.get('mil', {}) if isinstance(cfg.train, dict) else {}
    mil_cfg_model = cfg.model.get('mil', {}) if isinstance(cfg.model, dict) else {}

    time_aware = mil_cfg_train.get('time_aware', mil_cfg_model.get('time_aware', None))

    tau_from_cfg = mil_cfg_train.get('temperature', mil_tau_legacy if mil_tau_legacy is not None else 1.0)
    learnable_tau = bool(mil_cfg_train.get('learnable_tau', False))
    activation = str(mil_cfg_train.get('activation', mil_cfg_model.get('activation', 'softmax'))).lower()
    mil_tau = float(tau_from_cfg)

    head = build_mil_head(
        mil_variant,
        cfg.model['d_model'],
        n_classes,
        topk=topk_k,
        tau=mil_tau,
        learnable_tau=learnable_tau,
        activation=activation,
        time_aware=time_aware,
    ).to(device)
    ta_msg = ''
    if isinstance(time_aware, dict) and time_aware.get('enable', False):
        ta_mode = str(time_aware.get('mode', 'relbias'))
        ta_msg = f", time_aware=on(mode={ta_mode})"
    else:
        ta_msg = ", time_aware=off"
    logger.info(f"MIL variant: {mil_variant} (topk={topk_k}, tau={mil_tau}, learnable_tau={learnable_tau}, activation={activation}{ta_msg})")

    # --- ADR (Attention Diversity Regularization) ---
    adr_cfg = cfg.train.get('adr', {}) if isinstance(cfg.train, dict) else {}
    adr_enable = bool(adr_cfg.get('enable', False))
    adr_weight = float(adr_cfg.get('weight', 0.0))
    if adr_enable:
        logger.info(f'ADR enabled: weight={adr_weight}')

    def _attention_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute mean entropy over bag dimension for attention weights.
        attn: [B, P] or [B, P, 1]. Returns scalar tensor.
        """
        if attn.dim() == 3:
            attn = attn.squeeze(-1)
        attn = attn.clamp_min(eps)
        attn = attn / attn.sum(dim=1, keepdim=True)
        H = -(attn * attn.log()).sum(dim=1)
        return H.mean()

    opt = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': cfg.train['backbone_lr']},
        {'params': head.parameters(), 'lr': cfg.train['head_lr']}
    ], weight_decay=cfg.train.get('weight_decay', 0.05))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    logger.info('Scheduler: ReduceLROnPlateau')

    label_smoothing = float(cfg.train.get('label_smoothing', 0.0))
    max_grad_norm   = float(cfg.train.get('max_grad_norm', 1.0))

    patience_es = cfg.train.get('early_stop_patience', 10)
    es_counter = 0

    def _save_confusion(cm: np.ndarray, path: str):
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    best_f1 = -1.0
    for epoch in range(cfg.train['epochs']):
        model.train(); head.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            H = model(x)
            logits, attn = head(H)
            ce = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=label_smoothing)
            if adr_enable:
                H_attn = _attention_entropy(attn)
                loss = ce + adr_weight * (-H_attn)  # maximize attention entropy
            else:
                loss = ce
            opt.zero_grad(); loss.backward();
            clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_grad_norm)
            opt.step()
        # eval (on validation)
        model.eval(); head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                logits, _ = head(model(x))
                pred = logits.argmax(dim=1).cpu()
                ys.append(y); ps.append(pred)
        y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
        m = compute_metrics(y_true, y_pred)
        logger.info(f'Epoch {epoch+1}: val_acc={m["accuracy"]:.4f} val_f1={m["macro_f1"]:.4f}')

        # LR scheduler step
        scheduler.step(m['macro_f1'])

        # Per-class F1 and confusion matrix (val)
        f1_per = f1_score(y_true, y_pred, average=None, labels=classes)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Early stopping & best checkpoint
        improved = m['macro_f1'] > best_f1
        logger.info(f'val per-class F1: {f1_per}')
        if improved:
            best_f1 = m['macro_f1']
            es_counter = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({'model': model.state_dict(), 'head': head.state_dict()}, 'checkpoints/cls_millet.ckpt')
            logger.info('Checkpoint saved: checkpoints/cls_millet.ckpt')
            # save visualizations for the best model (val)
            _save_confusion(cm, 'reports/confusion_matrix_val_best.png')
            np.savetxt('reports/f1_per_class_val_best.csv', np.column_stack([classes, f1_per]), delimiter=',', header='class,f1', comments='')
        else:
            es_counter += 1
            if es_counter >= patience_es:
                logger.info(f'Early stopping triggered (patience={patience_es}). Best val macro-F1={best_f1:.4f}')
                break

    # Final evaluation on test set using the best checkpoint
    logger.info('Evaluating best checkpoint on test set...')
    ckpt = torch.load('checkpoints/cls_millet.ckpt', map_location='cpu')
    model.load_state_dict(ckpt['model']); head.load_state_dict(ckpt['head'])
    model.to(device); head.to(device)

    model.eval(); head.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits, _ = head(model(x))
            pred = logits.argmax(dim=1).cpu()
            ys.append(y); ps.append(pred)
    y_true = torch.cat(ys).numpy(); y_pred = torch.cat(ps).numpy()
    m = compute_metrics(y_true, y_pred)
    logger.info(f'TEST: acc={m["accuracy"]:.4f} f1={m["macro_f1"]:.4f}')
    f1_per = f1_score(y_true, y_pred, average=None, labels=classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    _save_confusion(cm, 'reports/confusion_matrix_test.png')
    np.savetxt('reports/f1_per_class_test.csv', np.column_stack([classes, f1_per]), delimiter=',', header='class,f1', comments='')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--pretrained', type=str, default='checkpoints/ssl_ppt.ckpt')
    args = ap.parse_args()
    cfg = Config.load(args.config)
    # non-destructive overrides; allow CLI to set/override
    if not hasattr(cfg, 'data_root') or not cfg.data_root:
        cfg.data_root = args.data_root
    else:
        if args.data_root:
            cfg.data_root = args.data_root
    if not hasattr(cfg, 'train') or cfg.train is None:
        cfg.train = {}
    if args.pretrained:
        cfg.train['pretrained'] = args.pretrained
    train_eval(cfg)