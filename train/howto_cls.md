<file name=1 path=/Users/jld20024/Desktop/python/time-series-ssl/train/howto_cls.md># Time-Series Classification with MILLET

## 背景
自己教師あり学習 (SSL) フェーズで HAR データを用いて学習した表現 (PatchTST + PPT) を活用し、
実際のラベル付きデータセット (UWaveGestureLibrary) を用いた分類を行います。
このステップが **下流タスク (downstream task)** であり、本来の目的であるジェスチャ認識を実現します。

採用論文：
- **MILLET: Multiple Instance Learning Transformer for Time-Series (AAAI 2025)**

## 学習の中身

### 1. 入力データ
- UWaveGestureLibrary (Gesture) の `.pt` データ (train/test/val)。
- フォーマットは `[N, T, C]` で、系列長 206、チャネル数 3。

### 2. PatchTST バックボーン
- SSL で学習済みの重み (`ssl_ppt.ckpt`) をロード。
- 入力系列をパッチに分割し Transformer Encoder で特徴表現 `[B, P, D]` を得る。

### 3. MILLET ヘッド (Attention-based MIL)
- 各パッチ特徴 `[B, P, D]` に対し、アテンション機構で重要度を推定。
- 重み付き平均を計算し、サンプルごとの特徴ベクトル `[B, D]` を抽出。
- 全結合層でクラス分類 (例: 8クラスのジェスチャ)。

#### 温度付きアテンション (Temperature-scaled Attention)
- アテンション分布に **温度 (τ)** を導入することで、重みのシャープネスを制御。
- 数式的には softmax(QK^T / τ) を計算する形になり、τ < 1 でよりシャープに、τ > 1 でよりスムーズに分布。
- この調整により、特定パッチへの過集中や分散をコントロール可能となり、クラスごとの識別性能の改善に寄与。
- 本実装では τ をハイパーパラメータとして設定し、必要に応じて学習可能変数とすることも可能。

#### 本実装での精度改善施策、採用状況
- **採用**：温度付きアテンション（τ=1.1, 固定）。注意の集中度を緩和して全体精度の安定化に寄与。
- **非採用**：ADR（Attention Diversity Regularization）。当データ規模では精度改善が限定的で一部指標で悪化したためオフにしています（実装は残し、`train.adr.enable=false`）。
- **非採用（今回の最終版ではオフ）**：entmax / time-aware pooling / TTA / EMA / SWA。副作用や効果が限定的だったため、本番想定のシンプルで再現性の高い構成を優先。

### 4. 学習の流れ
- 損失関数: CrossEntropyLoss。
- Optimizer: AdamW (backbone に低学習率、head に高学習率を設定)。
- Epochごとに train/test を繰り返し、Accuracy / Macro-F1 を記録。
- ベスト性能更新ごとに checkpoint を保存 (`cls_millet.ckpt`)。

## 狙い・目的
- SSL で得た表現を **転移学習**し、分類性能を最大化する。
- 特にデータ数が少ない場合でも、SSL で学習した表現が汎化性能を向上させることを期待。

## まとめ
- `train_cls.py`は **自己教師あり学習で獲得した特徴を実際の分類に活用するプロセス**。
- PatchTST により時系列をパッチ化し、MILLET により「重要なパッチを強調」して分類。
- 狙いは **ジェスチャ認識を高精度に行う**ことであり、モデルによるジェスチャー分類を実現するコア部分。