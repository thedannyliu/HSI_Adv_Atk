# HyperForensics++ 高光譜影像偽造檢測與對抗攻擊框架

HyperForensics++ 是一個全面的高光譜影像偽造檢測與對抗攻擊研究框架，旨在提供一個系統化的環境，用於評估偽造檢測模型的魯棒性，以及研究和開發對抗攻擊方法。

## 功能特點

- **模型訓練與評估**：提供完整的模型訓練和評估流程
- **多種對抗攻擊方法**：實現多種對抗攻擊算法 (FGSM, PGD, CW, DeepFool 等)
- **域特定攻擊**：專門針對空間域和光譜域的攻擊實現
- **全面的評估指標**：提供詳細的分割評估和對抗攻擊評估指標
- **高光譜數據處理**：專門針對高光譜影像的數據加載和預處理
- **可視化工具**：豐富的可視化功能，幫助理解檢測結果和攻擊效果

## 目錄結構

```
HyperForensics++/
├── config/             # 配置文件
├── models/             # 模型定義
├── datasets/           # 數據集和數據加載
├── attacks/            # 對抗攻擊實現
│   ├── spatial_attacks.py    # 空間域攻擊
│   ├── spectral_attacks.py   # 光譜域攻擊
│   └── utils.py              # 攻擊工具函數
├── utils/              # 工具函數
├── metrics/            # 評估指標
├── scripts/            # 腳本工具
└── output/             # 輸出結果目錄
```

## 快速開始

### 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt
```

### 訓練模型

```bash
# 使用所有配置訓練
python scripts/train.py --config_path ./config/train_config.yaml

# 使用特定配置訓練（例如只使用 config1）
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# 使用多個特定配置訓練
python scripts/train.py --config_path ./config/train_config.yaml --configs config1 config2

# 從檢查點繼續訓練
python scripts/train.py --config_path ./config/train_config.yaml --configs config1 --checkpoint ./logs/weights/exp_20240320_120000/checkpoint_epoch_50.pth
```

注意：
- 使用 `--configs` 參數時，程式會自動包含 "Origin" 配置（用於驗證集）
- 您也可以直接修改 `train_config.yaml` 中的 `CONFIGS` 列表來選擇要使用的配置
- 訓練過程會自動保存以下權重：
  - `best_miou.pth`：最佳 mIoU 的模型權重
  - `best_acc.pth`：最佳準確率的模型權重
  - `last.pth`：最後一個 epoch 的模型權重
  - `checkpoint_epoch_X.pth`：每5個 epoch 的檢查點

### 遠端訓練

為了避免遠端訓練因斷線而中斷，您可以使用以下方法之一：

1. 使用 `screen`（推薦）：
```bash
# 創建新的 screen 會話
screen -S training

# 在 screen 會話中運行訓練腳本
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# 按 Ctrl+A 然後按 D 來分離 screen 會話
# 使用 screen -ls 查看所有會話
# 使用 screen -r training 重新連接會話
```

2. 使用 `tmux`：
```bash
# 創建新的 tmux 會話
tmux new -s training

# 在 tmux 會話中運行訓練腳本
python scripts/train.py --config_path ./config/train_config.yaml --configs config1

# 按 Ctrl+B 然後按 D 來分離 tmux 會話
# 使用 tmux ls 查看所有會話
# 使用 tmux attach -t training 重新連接會話
```

3. 使用 `nohup`：
```bash
nohup python scripts/train.py --config_path ./config/train_config.yaml --configs config1 > training.log 2>&1 &
```

所有訓練日誌都會保存在 `/ssd1/dannyliu/HSI_Adv_Atk/Log` 目錄下，文件名格式為 `exp_{timestamp}_lr{learning_rate}_bs{batch_size}.log`。

### 生成對抗樣本

```bash
# 空間域攻擊
python scripts/generate_adversarial.py --attack pgd --domain spatial

# 光譜域攻擊
python scripts/generate_adversarial.py --attack fgsm --domain spectral

# 混合域攻擊
python scripts/generate_adversarial.py --attack pgd --domain hybrid
```

### 評估性能

```bash
# 評估原始數據集
python scripts/evaluate.py --config_path ./config/test_config.yaml --mode original

# 評估對抗樣本
python scripts/evaluate.py --config_path ./config/test_config.yaml --mode adversarial --adv_dir ./output/adversarial_samples/spatial_PGD_eps0.05_20230320_120000/samples --adv_flist ./output/adversarial_samples/spatial_PGD_eps0.05_20230320_120000/adv_filelist.txt
```

## 支持的攻擊方法

### 基本方法
- **FGSM** (Fast Gradient Sign Method)：快速梯度符號法
- **PGD** (Projected Gradient Descent)：投影梯度下降法
- **CW** (Carlini-Wagner)：Carlini-Wagner 攻擊
- **DeepFool**：DeepFool 攻擊

### 攻擊域
- **空間域攻擊**：專注於修改影像的空間結構，對所有光譜波段施加一致的空間扰動模式
- **光譜域攻擊**：專注於修改特定波段的光譜特徵，保持空間結構相對不變
- **混合域攻擊**：結合空間域和光譜域的攻擊，同時在兩個維度上進行擾動

## 評估指標

### 分割評估指標
- 像素準確率 (Pixel Accuracy)
- 平均IoU (Mean IoU)
- 類別F1分數 (Class F1 Score)

### 對抗攻擊評估指標
- 攻擊成功率 (Attack Success Rate)
- 擾動范數 (L0, L1, L2, L∞)
- 信噪比 (Signal-to-Noise Ratio)
- 結構相似性 (Structural Similarity)

## 攻擊方法比較

| 攻擊方法 | 空間域 | 光譜域 | 混合域 | 計算複雜度 | 攻擊成功率 |
|---------|-------|-------|--------|----------|-----------|
| FGSM    | ✓     | ✓     | ✓      | 低       | 中        |
| PGD     | ✓     | ✓     | ✓      | 中       | 高        |
| CW      | ✓     | ✓     | ✓      | 高       | 高        |
| DeepFool| ✓     | ✓     | ✓      | 中       | 中        |

### 混合域攻擊優勢

混合域攻擊結合了空間域和光譜域攻擊的優點，通過同時在這兩個維度上進行擾動，能夠:

1. **更高攻擊成功率**: 同時利用空間和光譜兩個維度的脆弱性
2. **更難檢測**: 通過在兩個維度上分散擾動，使擾動更加隱蔽
3. **更強的泛化能力**: 對不同類型的防禦機制有更好的穿透能力

使用混合攻擊時，可通過配置文件調整空間域和光譜域的權重比例(`SPATIAL_WEIGHT`)，以及目標波段的選擇。

```yaml
# config/attack_config.yaml
ATTACK:
  # 其他參數...
  
  # 混合攻擊參數
  HYBRID:
    SPATIAL_WEIGHT: 0.5  # 空間域攻擊的權重 (0-1)
    TARGET_BANDS: null   # 要攻擊的特定波段，null表示自動選擇重要波段
```

### 自定義配置

您可以通過修改 `config/attack_config.yaml` 文件來調整攻擊參數:

## 引用

如果您使用了HyperForensics++框架，請考慮引用以下論文：

```
@article{your_paper,
  title={HyperForensics++: A Framework for Hyperspectral Image Forgery Detection and Adversarial Attacks},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
```

## 許可證

本專案採用 MIT 許可證 