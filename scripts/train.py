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

from config import get_cfg_defaults
from datasets import forgeryHSIDataset
from models.hrnet import HRNetV2
from metrics import SegMetrics
from utils import set_random_seed, save_checkpoint, load_checkpoint, get_datetime_str
from utils import DummyTransform, PairCompose, RandomFlip


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


def train_model(model, train_loader, val_loader, criterion, optimizer, cfg, device):
    """
    訓練模型

    Args:
        model: 模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        criterion: 損失函數
        optimizer: 優化器
        cfg: 配置
        device: 計算設備
    """
    # 創建實驗目錄
    exp_name = f"exp_{get_datetime_str()}_lr{cfg.TRAIN.LEARNING_RATE}_bs{cfg.TRAIN.BATCH_SIZE}"
    save_dir = os.path.join(cfg.TRAIN.SAVE_WEIGHT_PATH, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 設置日誌記錄器
    logger = setup_logger("/ssd1/dannyliu/HSI_Adv_Atk/Log/training_log", exp_name)
    logger.info(f"權重將保存在: {save_dir}")

    # 初始化最佳指標
    best_miou = 0.0
    best_acc = 0.0
    
    # 斷點續訓
    start_epoch = cfg.TRAIN.EPOCH_START
    if cfg.TRAIN.CHECKPOINT and os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT, map_location=device)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"從 epoch {start_epoch} 繼續訓練")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCH_END + 1):
        model.train()
        losses = []

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        logger.info(f"Epoch [{epoch}/{cfg.TRAIN.EPOCH_END}] - Loss: {avg_loss:.4f}")

        # 驗證模型
        val_score = evaluate_model(model, val_loader, device, n_classes=cfg.DATASET.NUM_CLASSES)
        cur_miou = val_score["Mean_IoU"]
        cur_acc = val_score["Mean_Accuracy"]
        logger.info(f"  [Val] mIoU={cur_miou:.4f}, Acc={cur_acc:.4f}")

        # 如果表現更好，更新最佳模型
        updated = False
        if cur_miou > best_miou:
            best_miou = cur_miou
            # 保存最佳 mIoU 模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(save_dir, "best_miou.pth"))
            updated = True

        if cur_acc > best_acc:
            best_acc = cur_acc
            # 保存最佳 Acc 模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(save_dir, "best_acc.pth"))
            updated = True

        if updated:
            logger.info(f"  ==> 更新最佳模型: best_mIoU={best_miou:.4f}, best_acc={best_acc:.4f}")

        # 定期保存檢查點
        if (epoch + 1) % 5 == 0:  # 每5個epoch保存一次
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            logger.info(f"保存檢查點: epoch_{epoch+1}")

    # 保存最後一個epoch的權重
    torch.save({
        'epoch': cfg.TRAIN.EPOCH_END,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_miou': best_miou,
        'best_acc': best_acc,
    }, os.path.join(save_dir, "last.pth"))
    logger.info("[INFO] 訓練完成")

    # 保存訓練結果摘要
    results = {
        "best_miou": best_miou,
        "best_acc": best_acc,
        "final_epoch": cfg.TRAIN.EPOCH_END
    }
    with open(os.path.join(save_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"訓練結果: {results}")


def evaluate_model(model, dataloader, device, n_classes=2):
    """
    評估模型

    Args:
        model: 模型
        dataloader: 數據加載器
        device: 計算設備
        n_classes: 類別數

    Returns:
        dict: 評估結果
    """
    model.eval()
    metric = SegMetrics(n_classes, device=device)
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs, _ = model(images)
            preds = outputs.argmax(dim=1)
            metric.update(labels, preds)
    score = metric.get_results()
    model.train()
    return score


def create_dataset(cfg, config_name, split='train', transform=None):
    """
    創建指定配置的數據集
    
    Args:
        cfg: 配置對象
        config_name: 配置名稱
        split: 數據集分割類型
        transform: 數據轉換
    """
    # 在配置列表中查找指定的配置
    config = next((c for c in cfg.DATASET.CONFIGS if c["name"] == config_name), None)
    if config is None:
        raise ValueError(f"找不到配置: {config_name}")
    
    # 根據分割類型選擇對應的文件列表
    if split == 'train':
        flist = os.path.join(cfg.DATASET.SPLIT_ROOT, config["train"])
    elif split == 'val':
        flist = os.path.join(cfg.DATASET.SPLIT_ROOT, config["val"])
    else:  # test
        flist = os.path.join(cfg.DATASET.SPLIT_ROOT, config["test"])
    
    return forgeryHSIDataset(
        root=cfg.DATASET.ROOT,
        flist=flist,
        split=split,
        target_type='mask',
        transform=transform,
        config=config_name
    )


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
        dataset = create_dataset(cfg, config["name"], 'train', train_transform)
        train_datasets.append(dataset)
        print(f"[INFO] 添加訓練數據集: {config['name']}, 大小: {len(dataset)}")
    
    # 合併所有訓練數據集
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=True
    )

    # 驗證集使用原始圖片
    val_dataset = create_dataset(cfg, "Origin", 'val', DummyTransform())
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        drop_last=False
    )

    # 建立模型
    model = HRNetV2(
        C=cfg.MODEL.C,
        num_class=cfg.DATASET.NUM_CLASSES,
        in_ch=cfg.MODEL.IN_CHANNELS,
        use3D=(cfg.MODEL.USE_3D == True),
        useAttention=(cfg.MODEL.USE_ATTENTION == True)
    )

    # 加載檢查點（如果有）
    if cfg.TRAIN.CHECKPOINT and os.path.isfile(cfg.TRAIN.CHECKPOINT):
        print(f"[INFO] 從 {cfg.TRAIN.CHECKPOINT} 加載檢查點")
        if cfg.TRAIN.CHECKPOINT.endswith('.pth'):
            # 只有模型權重
            model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT, map_location=device))
        else:
            # 完整檢查點
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=1e-5)
            start_epoch = load_checkpoint(model, optimizer, cfg.TRAIN.CHECKPOINT, device)

    model.to(device)

    # 優化器和損失函數
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 0.99]).to(device))

    # 訓練模型
    train_model(model, train_loader, val_loader, criterion, optimizer, cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練高光譜影像偽造檢測模型")
    parser.add_argument("--config_path", type=str, default="./config/train_config.yaml",
                      help="配置文件路徑")
    parser.add_argument("--configs", nargs="+", type=str,
                      help="要使用的配置名稱列表，例如：config1 config2")
    args = parser.parse_args()
    main(args) 