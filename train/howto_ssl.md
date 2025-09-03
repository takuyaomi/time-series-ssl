# Self-Supervised Learning (SSL) for Time-Series with PPT

## 背景
本プロジェクトでは UWaveGestureLibrary や HAR のようなセンサーデータを対象に、**自己教師あり学習 (Self-Supervised Learning; SSL)** を用いて表現学習を行い、その後に分類モデル (MILLET) を用いて時系列分類を行う流れを採用しています。

採用論文：
- **PPT: Pretext Tasks via Patch Permutation for Time-Series Representation Learning (ICLR 2024)**
- **MILLET: Multiple Instance Learning Transformer for Time-Series (AAAI 2025)**

## 学習の中身

### 1. 入力データ
- HAR データセットのセンサ信号（加速度など）を `[B, T, C]` 形式で読み込み。
- 例: バッチ128、系列長206、チャネル数3。

### 2. PatchTST バックボーン
- 系列をパッチ単位に分割し（例: 長さ32の窓をstride=16で切り出す）、Transformer Encoder でエンコード。
- 出力は `[B, P, D]` のパッチ表現（P=パッチ数, D=潜在次元）。

### 3. PPT (Pretext Task) ヘッド
- パッチ表現（PatchTSTの出力）に対して、**弱い順序シャッフル**（例: パッチの一部を軽度にシャッフル）と **強い順序シャッフル+ノイズ付与**（例: より多くのパッチをシャッフルし、ノイズも加える）を生成します。
- これらの変換を通じて、元の系列の情報がどれだけ保たれているかを学習します。
- 学習時には次の2種類の損失を同時に用います：
  - **InfoNCE コントラスト損失**：弱いシャッフルと強いシャッフルの特徴表現を「同じ系列の別ビュー」として近づけ、異なるサンプル間は遠ざけるようにします。
  - **Order Consistency 損失**：元系列・弱シャッフル・強シャッフルの順序が「元 > 弱 > 強」となるように順序保存性を学習します（系列の順序情報が失われていないかを監督）。
- **実装上は、各パッチ表現が両方の損失（InfoNCE, Order Consistency）を通じて同時に学習され、これらの損失を合計して最適化します。**

> **Note:** 実装では PatchTST バックボーンを用い、`patch_len=32`・`stride=16` でパッチ分割しています（この設定は事前学習・下流分類（MILLET）で一貫して固定する必要があります）。

> **Design choice rationale:** パッチ分割の長さやストライドを一貫して固定することで、SSLで学習したパッチ表現が下流のMILLETモデルでも安定して活用でき、転移時の表現崩壊や性能劣化を防ぎます。

### 4. 学習の進み方
- Epochごとに全バッチを処理。
- ログ出力：`ssl_loss` が大きく減少していく。
  - 例: Epoch1=2.0 → Epoch2=0.48 → Epoch3=0.046。
- Checkpoint (`checkpoints/ssl_ppt.ckpt`) がベスト更新のたび保存される。

## 狙い・目的
- **ラベルを使わずに** HAR の時系列から「順序」や「構造」を理解できる表現を獲得する。
- この事前学習済みのバックボーンを **UWaveGestureLibrary の分類タスクに転移**することで、少量データでも高精度に分類可能にする。

## まとめ
- `train_ssl.py`は **自己教師あり事前学習 (SSL)** フェーズの実行コード。
- やっていることは「系列をパッチに分けて順序を乱し、元の構造を学ばせる」こと。
- 狙いは **汎用的な特徴表現の獲得**であり、次のステップで MILLET 分類モデルに渡して実際のジェスチャ分類を行う。