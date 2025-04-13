#!/bin/bash

# 設置CUDA可見設備
export CUDA_VISIBLE_DEVICES=0

# 設置語言環境為英文
export LC_ALL=C

# 設置工作目錄
WORKSPACE_DIR="/ssd1/dannyliu/HSI_Adv_Atk/HyperForensics++"
cd $WORKSPACE_DIR

# 創建日誌文件夾
mkdir -p logs/training_log_dfcn_improved
mkdir -p logs/validation_results_dfcn_improved
mkdir -p checkpoints_dfcn_improved

# 顯示訓練開始信息
echo "=========================================="
echo "  Starting Improved HyperForensics++ Training"
echo "  $(date "+%A, %B %d, %Y %H:%M:%S")"
echo "=========================================="

# 運行訓練腳本，使用改進的配置文件
python scripts/train.py --config_path ./config/train_improved.yaml

# 顯示訓練完成信息
echo "=========================================="
echo "  Training completed"
echo "  $(date "+%A, %B %d, %Y %H:%M:%S")"
echo "==========================================" 