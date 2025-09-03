# Time-Series SSL + MILLET

## Overview
This repository implements a two-stage approach for time series gesture classification:
1. **Self-Supervised Learning (SSL) Pretraining** using PPT (Patch Order Do Matters).
2. **Supervised Classification** using MILLET (Multiple Instance Learning for Locally Explainable TSC).

The goal is to classify 8 gesture classes from accelerometer signals with high accuracy and robustness.

Detailed implementation notes:
- [train/howto_ssl.md](train/howto_ssl.md): SSL pretraining details (PPT).
- [train/howto_cls.md](train/howto_cls.md): Classification details (MILLET).

## Setup Instructions

### 1. Environment
Ensure Python **3.11.6** is installed. Then set up a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
Unzip the provided datasets into the correct directories:

```bash
unzip data/Gesture.zip -d data/gesture
unzip data/HAR.zip -d data/har
```

- **HAR** is used for SSL pretraining.
- **Gesture** is used for supervised classification.

### 3. Training

#### (1) Self-Supervised Pretraining (PPT)
```bash
python train/train_ssl.py --config configs/ssl_ppt.yaml --data-root data/har
```
This will produce a pretrained checkpoint:  
`checkpoints/ssl_ppt.ckpt`

#### (2) Classification (MILLET)
```bash
python train/train_cls.py --config configs/cls_millet.yaml --data_root data/gesture --pretrained checkpoints/ssl_ppt.ckpt
```
This will train MILLET on the gesture dataset using the pretrained backbone.

### 4. Reports
During training, the following are saved under `reports/`:
- **PNG plots**: learning curves, confusion matrices.
- **CSV files**: per-class metrics (F1, accuracy, etc).

## Results
- Best validation macro-F1 ≈ **0.8016**
- Test accuracy ≈ **0.8167**
- Class 4 remains the most challenging, with room for improvement.


## Notes
- All improvements, variants, and detailed explanations are described in `construction.md`.
- For in-depth technical details on training stages, consult:
  - [train/howto_ssl.md](train/howto_ssl.md)
  - [train/howto_cls.md](train/howto_cls.md)

## 実行手順一覧
```bash
# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# データの展開
unzip data/Gesture.zip -d data/gesture
unzip data/HAR.zip -d data/har

# SSL事前学習（PPT）
python train/train_ssl.py --config configs/ssl_ppt.yaml --data-root data/har

# 分類学習（MILLET）
python train/train_cls.py --config configs/cls_millet.yaml --data_root data/gesture --pretrained checkpoints/ssl_ppt.ckpt
```

