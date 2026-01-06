
<div align="center">

# 🦅 AOI Defect Classification: Champion's Solution
### 工業級瑕疵檢測系統 - 冠軍訓練方案

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![VRAM Optimization](https://img.shields.io/badge/VRAM-Optimized_for_24GB-success?style=for-the-badge)](config.py)
[![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)]()

**Combine Semi-Supervised Learning with High-Performance Ensemble**  
**專為 RTX 3090/4090 打造的神級訓練流程**

[Introduction](#-introduction-專案簡介) •
[Methodology](#-methodology-核心技術) •
[Installation](#-installation-安裝指南) •
[Pipeline](#-pipeline-執行流程) •
[Optimization](#-optimization-極速優化)

</div>

---

## 📖 Introduction (專案簡介)

本專案實現了一個基於 **Semi-Supervised Learning (半監督學習)** 的高精度 AOI 瑕疵檢測系統。我們的目標是利用 Pseudo Labeling (偽標籤) 技術，將 10,000 張未標註的測試集資料轉化為訓練資源，挑戰 **99.9%** 的分類準確率。

針對工業級應用場景，我們實作了多項 Kaggle Grandmaster 等級的技巧，解決了 **類別極度不平衡 (Class Imbalance)**、**雙流輸入的顯存瓶頸** 以及 **模型崩潰 (Mode Collapse)** 等關鍵痛點。

<div align="center">
  <img src="assets/overall_architecture.png" alt="Overall Architecture Diagram" width="95%">
</div>

---

---

## 🏆 Strategic Solutions (核心解題策略)

我們針對工業場景的三大痛點，提出了具體的技術解決方案：

### 1. 破解類別極度不平衡 (Solving Class Imbalance) ⚖️

**痛點**: "Horizontal Defect" (Label 2) 僅有 **100 張** (3.9%)，而 "Normal" 有 674 張。模型極易忽略稀有瑕疵，導致漏檢 (False Negative)。

**解決方案**:
*   **Weighted Random Sampler (加權隨機採樣)**:
    *   我們不使用標準採樣，而是賦予稀有類別極高的權重。
    *   **機制**: 確保在一個 Epoch 中，模型看到 "Label 2" 的次數與 "Label 0" 一樣多。這等同於對稀有瑕疵進行了 **6.7倍** 的過採樣 (Oversampling)。
*   **Threshold Optimization (閾值優化)**:
    *   傳統 Argmax (0.5) 對稀有類別不利。我們在推論階段對 Label 2 實施 **Aggressive Recall** 策略。
    *   若 Label 2 的預測機率 > **0.4** (而非 0.5)，即強制判定為瑕疵，大幅降低漏殺率。

### 2. 突破顯存瓶頸 (Overcoming VRAM Limits) 💾

**痛點**: 傳統 "Dual Stream" (雙流) 網路需要同時輸入 "原圖" + "銳化圖"，顯存佔用翻倍 (2x VRAM)，導致無法在 RTX 3090/4090 上訓練 Large 模型。

**解決方案: Dual Stream Simulation (雙流時域模擬)**
我們利用 **時間軸 (Temporal Axis)** 來模擬雙流。不再同時輸入兩張圖，而是在 `dataset.py` 中設置 **Augmentation Switch**：

<div align="center">
  <img src="assets/dual_stream_sim.png" alt="Dual Stream Simulation Diagram" width="85%">
</div>

*   **運作原理**:
    *   **Epoch N**: 模型有 30% 機率看到 **模糊 (Blur)** 的影像 -> 強迫學習 **形狀 (Shape)** 特徵。
    *   **Epoch N+1**: 模型有 30% 機率看到 **銳化 (Sharpen)** 的影像 -> 強迫學習 **紋理 (Texture)** 特徵。
*   **效益**: 單一模型 (Single Stream) 卻擁有了雙流模型的魯棒性，且 **VRAM 零增加**。

### 3. 防止模型崩潰 (Preventing Mode Collapse) 📉

**痛點**: Swin Transformer Large 在訓練初期 (Warmup) 極不穩定，梯度容易瞬間爆炸 (Gradient Explosion)，導致 Loss 卡在 1.79 (Mode Collapse)，預測全變為同一類。

**解決方案**:
*   **Gradient Clipping (梯度剪裁)**:
    *   在 Backpropagation 之後、Optimizer Update 之前，強制將梯度的 Norm 限制在 **1.0** 以內。
    *   `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
    *   這就像"保險絲"，確保即使遇到極端數據，權重也不會被炸飛。
*   **3-Epoch Warmup**:
    *   前 3 個 Epoch 學習率從 `1e-6` 徐緩升至 `1e-4`，讓 Pretrained Weights 能夠適應新數據的分佈。

---

---

## ⚙️ Detailed Configuration Analysis (參數深度解析)

為了讓您完全掌控訓練細節，我們在此公開所有關鍵參數的設定邏輯與數值。所有設定皆位於 `config.py`。

### 1. Training Dynamics (訓練動態)

| 參數 (Parameter) | 數值 (Value) | 設計邏輯 (Rationale) |
| :--- | :--- | :--- |
| **`EPOCHS`** | 20 | 根據經驗，Swin/ConvNeXt 在 2,500 張圖上通常在 15-20 Epochs 收斂。過多容易 Overfit。 |
| **`WARMUP_EPOCHS`** | 3 | **關鍵設定**。前 3 個 Epochs 將 LR 從 1e-6 線性升至 1e-4。這是為了讓原本在 ImageNet 上預訓練好的權重 (Pretrained Weights) 能夠 "溫和地" 適應新的工業數據，避免一開始梯度過大破壞特徵提取層。 |
| **`LEARNING_RATE`** | 1e-4 | 配合 Cosine Annealing 策略。1e-4 是 Transformer 類模型 Fine-tuning 的黃金起點 (比 CNN 常用的 1e-3 低一個量級)。 |
| **`EMA_DECAY`** | 0.995 | 針對小數據集 (Small Data) 的特殊調整。標準 ImageNet 訓練通常用 0.9999，但在只有 2k 張圖的情況下，權重更新太慢會導致 EMA 跟不上，因此降至 0.995 加快收斂。 |
| **`WEIGHT_DECAY`** | 1e-2 | AdamW 的標準權重衰減，防止 Overfit。 |
| **`MAX_NORM`** | 1.0 | **Gradient Clipping** 閥值。Swin Transformer 對梯度非常敏感，設為 1.0 是防止訓練中途 Loss 突然炸裂 (Spike) 的保險絲。 |

### 2. Augmentation Hyperparameters (增強參數細節)

我們在 `dataset.py` 中定義了極致的增強策略。以下是 **Teacher Mode** 的具體參數，旨在模擬真實工業場景變異：

| 增強手法 (Technique) | 機率 (p) | 強度/參數 (Magnitude) | 作用 (Impact) |
| :--- | :--- | :--- | :--- |
| **Affine (仿射變換)** | 0.5 | Rotate ±30°, Scale 0.85-1.15 | 模擬輸送帶上工件的歪斜與遠近縮放。 |
| **Dual Stream Sim** | 0.3 | Blur (Limit 3) vs Sharpen (Alpha 0.2-0.5) | **核心技術**。隨機模擬失焦 (Defocus) 或過度銳化 (Artifacts) 的成像品質，強迫模型學習魯棒特徵。 |
| **CoarseDropout** | 0.5 | Holes=8, Size=Image//10 | 模擬局部遮擋或污漬，迫使模型看"整體"而非"局部"。 |
| **HSV / Brightness** | 0.5 | Shift Limit 0.2 | 模擬光源變化與不同機台的色差。 |

---

## 🏗️ Model Architecture Decisions (架構決策)

為什麼選擇這三口劍？

1.  **ConvNeXt V2 (Large)**:
    *   **角色**: 主力輸出 (Anchor)。
    *   **理由**: 結合了 CNN 的歸納偏置 (Inductive Bias) 與 Transformer 的訓練策略。對於 "紋理型" 瑕疵 (如刮痕) 檢測能力最強。
2.  **Swin Transformer V2 (Large)**:
    *   **角色**: 互補專家 (Complementary Expert)。
    *   **理由**: Window Attention 機制能捕捉長距離依賴。對於 "大面積" 或 "結構性" 瑕疵 (如大塊污漬) 表現優於 CNN。
3.  **EVA-02 (Large / MIM)**:
    *   **角色**: 穩健特徵 (Robustness)。
    *   **理由**: 基於 MIM (Masked Image Modeling) 預訓練，對抗噪聲與遮擋的能力極強，能修正在極端增強下的誤判。

---

## 🛠 Installation (安裝指南)

### 1. 環境設定
建議使用 Mamba/Conda 建立環境：
```bash
conda create -n aoi python=3.10
conda activate aoi
pip install -r requirements.txt
```

### 2. 資料結構
請將 `aoi_data.zip` 解壓至專案根目錄下的 `data/`：
```text
data/
├── train_images/  (2,528 images)
├── test_images/   (10,142 images)
├── train.csv
└── test.csv
```

---

## 🚀 Pipeline (執行流程)

您可以執行 `run_pipeline.sh` 一鍵完成，或參考以下步驟手動執行。

### Step 1: Teacher Model Training

這是整個 Pipeline 的基石。我們使用 ImageNet Pretrained 模型進行遷移學習。

<div align="center">
  <img src="assets/teacher_flow.png" alt="Teacher Training Flow" width="60%">
</div>

```bash
# 訓練 Teacher 模型 (支援 convnext, swinv2, eva02)
python train_teacher.py --model convnext

# 訓練 Swin V2 (可選)
python train_teacher.py --model swinv2
```

### Step 2: Pseudo Labeling (產生偽標籤)
```bash
# 產生 train_pseudo.csv
python inference_pseudo.py
```
> **Note**: 此步驟會自動載入所有訓練好的模型進行與集成。

### Step 3: Student Model Training

利用擴增後的數據集進行 **Noisy Student Training**。

<div align="center">
  <img src="assets/student_flow.png" alt="Student Training Flow" width="60%">
</div>

```bash
# 訓練 Student 模型 (讀取 train_pseudo.csv)
python train_student.py --model convnext
```

### Step 4: 冠軍推論 (Multi-Model Champion Ensemble)

這是最終的集成腳本，支援 **多模型加權投票 (Weighted Voting)**。

<div align="center">
  <img src="assets/code_flow_diagram.png" alt="Inference Flow" width="70%">
</div>

```bash
python ensemble_inference.py --output submission.csv
```

#### 🧪 Quick Check (單一模型快速驗證)
如果您剛訓練好一個模型 (例如 `convnext`) 想馬上看結果，不需要跑完整的集成，可以使用 `--model` 參數：

```bash
# 僅使用 ConvNeXt 進行推論與 Threshold Optimization
python ensemble_inference.py --model convnext --output submission_convnext.csv
```
> 這會自動執行該模型的 5-Fold Ensembling + 5-View TTA，並產出預測分佈供您檢查 (特別注意 Label 2 的數量)。

**如何調優**:
開啟 `ensemble_inference.py`，調整 `MODEL_WEIGHTS` 字典：
```python
MODEL_WEIGHTS = {
    'convnext': 0.50,  # 主力模型 (CV分數高)
    'swinv2':   0.30,  # 輔助模型 (Transformer 架構)
    'eva02':    0.20   # 輔助模型 (大尺寸)
}
```
*   建議給 Local CV 分數較高的模型更大的權重。

---

## 📊 Configuration

主要參數位於 `config.py`，可根據硬體調整：

```python
# config.py
BATCH_SIZE = 16        # 目標 Batch Size
CACHE_IMAGES = True    # 開啟 RAM Cache
USE_AMP = True         # 開啟混合精度
```

---

<div align="center">
    <p>Empowered by Advanced Agentic Coding</p>
</div>

---

## 📈 Performance & Results (實戰成效)

我們單獨使用 **ConvNeXt V2 Large** 進行訓練與測試，在不使用任何 Ensemble 的情況下即取得了驚人的成績。

### 🏅 Leaderboard Ranking
*   **Rank**: **9 / 969** (Top 1%) 
*   **Score**: **0.9972872**
*   **Model**: Single ConvNeXt V2 (5-Fold CV)

<div align="center">
  <img src="assets/convnext_accuracy_chart.png" alt="Validation Accuracy" width="48%">
  <img src="assets/convnext_loss_chart.png" alt="Training Loss" width="48%">
</div>

> **Note**: 訓練曲線顯示 Fold 2 與 Fold 3 的 Validation Accuracy 甚至達到了 **100%**，證明了我們解決方案的強大魯棒性。

---

## 📚 References (參考文獻)

1.  **ConvNeXt V2**: [Woo, S., et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." (2023)](https://arxiv.org/abs/2301.00808)
2.  **Swin Transformer V2**: [Liu, Z., et al. "Swin Transformer V2: Scaling Up Capacity and Resolution." (2022)](https://arxiv.org/abs/2111.09883)
3.  **EVA-02**: [Fang, Y., et al. "EVA-02: A Visual Representation for Neon Genesis." (2023)](https://arxiv.org/abs/2303.11331)
4.  **Noisy Student**: [Xie, Q., et al. "Self-training with Noisy Student improves ImageNet classification." (2020)](https://arxiv.org/abs/1911.04252)
5.  **Mean Teacher (EMA)**: [Tarvainen, A., & Valpola, H. "Mean teachers are better role models." (2017)](https://arxiv.org/abs/1703.01780)
6.  **Albumentations**: [Buslaev, A., et al. "Albumentations: Fast and Flexible Image Augmentations." (2020)](https://github.com/albumentations-team/albumentations)
7.  **AdamW (Decoupled Weight Decay)**: [Loshchilov, I., & Hutter, F. "Decoupled Weight Decay Regularization." (2017)](https://arxiv.org/abs/1711.05101)
8.  **SGDR (Cosine Annealing)**: [Loshchilov, I., & Hutter, F. "SGDR: Stochastic Gradient Descent with Warm Restarts." (2016)](https://arxiv.org/abs/1608.03983)
9.  **Gradient Clipping**: [Pascanu, R., et al. "On the difficulty of training recurrent neural networks." (2013)](https://arxiv.org/abs/1211.5063)
10. **timm (PyTorch Image Models)**: [Wightman, R. "PyTorch Image Models." (2019)](https://github.com/rwightman/pytorch-image-models)
11. **Cutout (CoarseDropout)**: [DeVries, T., & Taylor, G. W. "Improved Regularization of Convolutional Neural Networks with Cutout." (2017)](https://arxiv.org/abs/1708.04552)
