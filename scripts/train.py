import os
import sys
import logging
from datetime import datetime
# 添加專案根目錄到Python路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import numpy as np
import json
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml
from pathlib import Path
import torch.nn.functional as F

from config import get_cfg_defaults
from datasets import forgeryHSIDataset
from models.hrnet import HRNetV2
from metrics import SegMetrics
from utils import set_random_seed, save_checkpoint, load_checkpoint, get_datetime_str
from utils import DummyTransform, PairCompose, RandomFlip
from utils.metrics import calculate_miou, calculate_accuracy


def setup_logger(log_dir, exp_name):
    """
    設置日誌記錄器
    
    Args:
        log_dir: 日誌目錄
        exp_name: 實驗名稱
    """
    # 創建日誌目錄
    os.makedirs(log_dir, exist_ok=True)
    
    # 設置日誌文件名
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    
    # 配置日誌記錄器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(cfg, split='train', config_name=None):
    """創建數據集"""
    if config_name is None:
        # 如果沒有指定配置，使用所有配置
        datasets = []
        for config in cfg['DATASET']['CONFIGS']:
            dataset = forgeryHSIDataset(
                root=cfg['DATASET']['ROOT'],
                flist=config[split],
                split=split,
                config=config['name']
            )
            datasets.append(dataset)
        return ConcatDataset(datasets)
    else:
        # 使用指定的配置
        config = next((c for c in cfg['DATASET']['CONFIGS'] if c['name'] == config_name), None)
        if config is None:
            raise ValueError(f"找不到配置：{config_name}")
        return forgeryHSIDataset(
            root=cfg['DATASET']['ROOT'],
            flist=config[split],
            split=split,
            config=config_name
        )


def train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    total_miou = 0
    total_acc = 0
    
    # 用於監控預測分佈
    pred_ones_percentage = 0
    batch_count = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, targets, _) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device).long()  # 確保 targets 是 long 類型
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 處理模型輸出可能是 tuple 的情況
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 取第一個輸出
        
        # 檢查輸出形狀
        if outputs.dim() == 4:
            if outputs.size(1) == 2:  # (B, 2, H, W) 形式，即多類別分類
                loss = criterion(outputs, targets)
                # 使用 softmax 獲取概率，然後取 argmax 獲取預測類別
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1)
            else:  # (B, 1, H, W) 形式，即二元分類
                outputs = outputs.squeeze(1)  # 變為 (B, H, W)
                loss = criterion(outputs, targets)
                # 使用 sigmoid 獲取概率，然後閾值處理
                pred = (torch.sigmoid(outputs) > 0.5).long()
        else:
            # 單通道輸出
            loss = criterion(outputs, targets)
            pred = (outputs > 0).long()
        
        loss.backward()
        optimizer.step()
        
        # 監控預測中 1 的比例
        ones_percentage = pred.float().mean().item() * 100
        pred_ones_percentage += ones_percentage
        batch_count += 1
        
        # 計算指標 - 需確保 pred 和 targets 都是浮點型
        miou = calculate_miou(pred.float(), targets.float())
        acc = calculate_accuracy(pred.float(), targets.float())
        
        total_loss += loss.item()
        total_miou += miou
        total_acc += acc
        
        # 定期打印預測分佈
        if batch_idx % 20 == 0:
            target_ones = targets.float().mean().item() * 100
            logger.info(f"Batch {batch_idx} - 標籤中1的比例: {target_ones:.2f}%, 預測中1的比例: {ones_percentage:.2f}%")
        
        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'miou': f'{miou:.4f}',
            'acc': f'{acc:.4f}',
            'pred1%': f'{ones_percentage:.2f}%'
        })
    
    # 計算平均值
    avg_loss = total_loss / len(train_loader)
    avg_miou = total_miou / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_pred_ones = pred_ones_percentage / batch_count
    
    logger.info(f"Epoch Training - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}, Acc: {avg_acc:.4f}, Avg Pred 1%: {avg_pred_ones:.2f}%")
    return avg_loss, avg_miou, avg_acc


def validate(model, val_loader, criterion, device, logger):
    """驗證模型"""
    model.eval()
    total_loss = 0
    total_miou = 0
    total_acc = 0
    
    # 用於監控預測分佈
    pred_ones_percentage = 0
    batch_count = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device).long()  # 確保 targets 是 long 類型
            
            outputs = model(images)
            
            # 處理模型輸出可能是 tuple 的情況
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 取第一個輸出
            
            # 檢查輸出形狀
            if outputs.dim() == 4:
                if outputs.size(1) == 2:  # (B, 2, H, W) 形式，即多類別分類
                    loss = criterion(outputs, targets)
                    # 使用 softmax 獲取概率，然後取 argmax 獲取預測類別
                    probs = F.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1)
                else:  # (B, 1, H, W) 形式，即二元分類
                    outputs = outputs.squeeze(1)  # 變為 (B, H, W)
                    loss = criterion(outputs, targets)
                    # 使用 sigmoid 獲取概率，然後閾值處理
                    pred = (torch.sigmoid(outputs) > 0.5).long()
            else:
                # 單通道輸出
                loss = criterion(outputs, targets)
                pred = (outputs > 0).long()
            
            # 監控預測中 1 的比例
            ones_percentage = pred.float().mean().item() * 100
            pred_ones_percentage += ones_percentage
            batch_count += 1
            
            # 計算指標 - 需確保 pred 和 targets 都是浮點型
            miou = calculate_miou(pred.float(), targets.float())
            acc = calculate_accuracy(pred.float(), targets.float())
            
            total_loss += loss.item()
            total_miou += miou
            total_acc += acc
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'miou': f'{miou:.4f}',
                'acc': f'{acc:.4f}',
                'pred1%': f'{ones_percentage:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_miou = total_miou / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    avg_pred_ones = pred_ones_percentage / batch_count
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}, Acc: {avg_acc:.4f}, Avg Pred 1%: {avg_pred_ones:.2f}%")
    return avg_loss, avg_miou, avg_acc


def main(args):
    # 讀取配置
    cfg = get_cfg_defaults()
    if args.config_path:
        cfg.merge_from_file(args.config_path)
    
    # 如果指定了配置，則只使用指定的配置
    if args.configs:
        # 確保始終包含 Origin 配置（用於驗證）
        selected_configs = ["Origin"] + [c for c in args.configs if c != "Origin"]
        # 過濾配置列表，只保留選定的配置
        cfg.DATASET.CONFIGS = [c for c in cfg.DATASET.CONFIGS if c["name"] in selected_configs]
        print(f"[INFO] 使用選定的配置: {', '.join(selected_configs)}")
    
    cfg.freeze()

    # 設置設備
    if cfg.CUDA.USE_CUDA and torch.cuda.is_available():
        device_ids = cfg.CUDA.CUDA_NUM
        device_str = f"cuda:{device_ids[0]}"
        print(f"[INFO] 使用 GPU {device_ids[0]} ({torch.cuda.get_device_name(device_ids[0])})")
    else:
        device_str = 'cpu'
        print("[WARNING] 使用 CPU")
    device = torch.device(device_str)
    print(f"[INFO] 使用設備: {device_str}")
    set_random_seed(cfg.GENERAL.SEED)

    # 創建日誌和模型保存目錄
    os.makedirs(cfg.TRAIN.LOG_PATH, exist_ok=True)
    os.makedirs(cfg.TRAIN.SAVE_WEIGHT_PATH, exist_ok=True)

    # 訓練集使用數據增強
    train_transform = PairCompose([
        RandomFlip(p=0.5)
    ])
    
    # 創建所有配置的訓練數據集
    train_datasets = []
    for config in cfg.DATASET.CONFIGS:
        dataset = create_dataset(cfg, 'train', config["name"])
        train_datasets.append(dataset)
        print(f"[INFO] 添加訓練數據集: {config['name']}, 大小: {len(dataset)}")
    
    # 合併所有訓練數據集
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY
    )

    # 驗證集使用原始圖片
    val_dataset = create_dataset(cfg, 'val', 'Origin')
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.TEST.PIN_MEMORY
    )

    # 建立模型
    model = HRNetV2(
        C=cfg.MODEL.C,
        num_class=cfg.DATASET.NUM_CLASSES,
        in_ch=cfg.MODEL.IN_CHANNELS,
        use3D=(cfg.MODEL.USE_3D == True),
        useAttention=(cfg.MODEL.USE_ATTENTION == True)
    ).to(device)

    # 加載檢查點（如果有）
    if cfg.TRAIN.CHECKPOINT and os.path.isfile(cfg.TRAIN.CHECKPOINT):
        print(f"[INFO] 從 {cfg.TRAIN.CHECKPOINT} 加載檢查點")
        if cfg.TRAIN.CHECKPOINT.endswith('.pth'):
            # 只有模型權重
            model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT, map_location=device))
        else:
            # 完整檢查點
            optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            start_epoch = load_checkpoint(model, optimizer, cfg.TRAIN.CHECKPOINT, device)

    # 定義損失函數 - 調整權重以解決類別不平衡
    if cfg.DATASET.NUM_CLASSES == 2:  # 二分類任務
        # 計算類別權重 - 根據偽造像素佔比約為1-2%設置
        # 偽造類別(1)的權重是背景類別(0)的50倍
        class_weights = torch.tensor([1.0, 50.0]).to(device)
        print(f"[INFO] 設置類別權重: {class_weights.tolist()}, 重點關注偽造區域")
    else:
        # 使用配置中的權重
        class_weights = torch.tensor(cfg.TRAIN.LOSS.WEIGHTS).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[INFO] 使用 CrossEntropyLoss 損失函數，權重: {class_weights.tolist()}")
    
    # 定義優化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=(cfg.TRAIN.OPTIMIZER.BETA1, cfg.TRAIN.OPTIMIZER.BETA2)
    )
    
    # 定義學習率調度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg.TRAIN.SCHEDULER.MODE,
        factor=cfg.TRAIN.SCHEDULER.FACTOR,
        patience=cfg.TRAIN.SCHEDULER.PATIENCE,
        verbose=cfg.TRAIN.SCHEDULER.VERBOSE
    )
    print("添加學習率調度器: ReduceLROnPlateau")

    # 訓練模型
    logger = setup_logger(cfg.TRAIN.LOG_PATH, f"exp_{get_datetime_str()}_lr{cfg.TRAIN.LEARNING_RATE}_bs{cfg.TRAIN.BATCH_SIZE}")
    logger.info(f"權重將保存在: {cfg.TRAIN.SAVE_WEIGHT_PATH}")

    # 訓練循環
    best_miou = 0
    best_acc = 0
    for epoch in range(cfg.TRAIN.EPOCH_START, cfg.TRAIN.EPOCH_END + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.TRAIN.EPOCH_END}")
        
        # 訓練
        train_loss, train_miou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # 驗證
        val_loss, val_miou, val_acc = validate(
            model, val_loader, criterion, device, logger
        )
        
        # 訓練循環中添加學習率調度器
        # 在每個 epoch 結束後更新學習率
        scheduler.step(val_miou)
        
        # 記錄指標
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            # 保存最佳 mIoU 模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(cfg.TRAIN.SAVE_WEIGHT_PATH, "best_miou.pth"))
            logger.info(f"保存最佳 mIoU 模型：{val_miou:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳 Acc 模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(cfg.TRAIN.SAVE_WEIGHT_PATH, "best_acc.pth"))
            logger.info(f"保存最佳準確率模型：{val_acc:.4f}")
        
        # 保存最後一個 epoch 的模型
        if epoch == cfg.TRAIN.EPOCH_END:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(cfg.TRAIN.SAVE_WEIGHT_PATH, "last.pth"))
            logger.info("保存最後一個 epoch 的模型")

    # 保存訓練結果摘要
    results = {
        "best_miou": best_miou,
        "best_acc": best_acc,
        "final_epoch": cfg.TRAIN.EPOCH_END
    }
    with open(os.path.join(cfg.TRAIN.SAVE_WEIGHT_PATH, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"訓練結果: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練高光譜影像偽造檢測模型")
    parser.add_argument("--config_path", type=str, default="./config/train_config.yaml",
                      help="配置文件路徑")
    parser.add_argument("--configs", nargs="+", type=str,
                      help="要使用的配置名稱列表，例如：config1 config2")
    args = parser.parse_args()
    main(args) 