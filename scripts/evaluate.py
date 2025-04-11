import torch
import argparse
import os
import json
import numpy as np
import yaml

from torch.utils.data import DataLoader
from config import get_cfg
from datasets import forgeryHSIDataset
from models.hrnet import HRNetV2
from metrics import SegMetrics, AdvMetrics
from utils import set_random_seed, DummyTransform, PairCompose
from utils import visualize_results, visualize_adversarial


def evaluate_on_dataset(model, dataloader, device, n_classes=2, save_dir=None):
    """
    評估模型在數據集上的表現
    
    Args:
        model: 模型
        dataloader: 數據加載器
        device: 計算設備
        n_classes: 類別數量
        save_dir: 可視化結果保存目錄
        
    Returns:
        dict: 評估結果
    """
    model.eval()
    metrics = SegMetrics(n_classes, device=device)
    
    with torch.no_grad():
        for batch_idx, (images, labels, filenames) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs, _ = model(images)
            preds = outputs.argmax(dim=1)
            
            # 更新評估指標
            metrics.update(labels, preds)
            
            # 保存可視化結果
            if save_dir and batch_idx % 10 == 0:  # 每10批保存一次
                for i in range(min(4, images.size(0))):  # 最多保存4個樣本
                    img_np = images[i].cpu().numpy()
                    label_np = labels[i].cpu().numpy()
                    pred_np = preds[i].cpu().numpy()
                    
                    vis_path = os.path.join(save_dir, f"{filenames[i].replace('.npy', '')}_vis.png")
                    visualize_results(img_np, label_np, pred_np, save_path=vis_path)
    
    # 獲取最終評估結果
    results = metrics.get_results()
    return results


def evaluate_adversarial(model, orig_dataloader, adv_dataloader, device, n_classes=2, save_dir=None):
    """
    評估模型在原始數據集和對抗數據集上的表現差異
    
    Args:
        model: 模型
        orig_dataloader: 原始數據加載器
        adv_dataloader: 對抗數據加載器
        device: 計算設備
        n_classes: 類別數量
        save_dir: 可視化結果保存目錄
        
    Returns:
        dict: 評估結果
    """
    model.eval()
    seg_metrics_orig = SegMetrics(n_classes, device=device)
    seg_metrics_adv = SegMetrics(n_classes, device=device)
    adv_metrics = AdvMetrics(device=device)
    
    # 確保兩個數據加載器的大小相同
    assert len(orig_dataloader) == len(adv_dataloader), "原始數據集和對抗數據集大小不一致"
    
    with torch.no_grad():
        for (orig_images, orig_labels, orig_filenames), (adv_images, adv_labels, adv_filenames) in zip(orig_dataloader, adv_dataloader):
            # 確保文件名匹配
            for of, af in zip(orig_filenames, adv_filenames):
                assert os.path.basename(of) == os.path.basename(af), f"文件名不匹配: {of} vs {af}"
            
            orig_images = orig_images.to(device)
            orig_labels = orig_labels.to(device, dtype=torch.long)
            adv_images = adv_images.to(device)
            adv_labels = adv_labels.to(device, dtype=torch.long)
            
            # 獲取原始和對抗預測
            orig_outputs, _ = model(orig_images)
            adv_outputs, _ = model(adv_images)
            
            orig_preds = orig_outputs.argmax(dim=1)
            adv_preds = adv_outputs.argmax(dim=1)
            
            # 更新評估指標
            seg_metrics_orig.update(orig_labels, orig_preds)
            seg_metrics_adv.update(adv_labels, adv_preds)
            
            # 更新對抗指標
            adv_metrics.update(orig_preds, adv_preds, orig_images, adv_images, orig_labels)
            
            # 保存可視化結果
            if save_dir:
                for i in range(min(2, orig_images.size(0))):  # 最多保存2個樣本
                    # 只有當原始預測和對抗預測不同時才保存
                    if (orig_preds[i] != adv_preds[i]).any():
                        img_name = os.path.basename(orig_filenames[i]).replace('.npy', '')
                        vis_path = os.path.join(save_dir, f"{img_name}_adv_compare.png")
                        
                        visualize_adversarial(
                            original=orig_images[i].cpu().numpy(),
                            adversarial=adv_images[i].cpu().numpy(),
                            original_pred=orig_preds[i].cpu().numpy(),
                            adversarial_pred=adv_preds[i].cpu().numpy(),
                            save_path=vis_path
                        )
    
    # 獲取最終評估結果
    orig_results = seg_metrics_orig.get_results()
    adv_results = seg_metrics_adv.get_results()
    adv_metrics_results = adv_metrics.get_results()
    
    # 合併結果
    results = {
        "Original": orig_results,
        "Adversarial": adv_results,
        "Attack_Metrics": adv_metrics_results
    }
    
    return results


def main(args):
    # 讀取配置
    cfg = get_cfg()
    if args.config_path:
        # 從文件加載配置
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            cfg.merge_from_dict(config_dict)
    cfg.freeze()
    
    # 設置設備
    device_ids = cfg.CUDA.CUDA_NUM
    device_str = f"cuda:{device_ids[0]}" if (cfg.CUDA.USE_CUDA and torch.cuda.is_available()) else 'cpu'
    device = torch.device(device_str)
    print(f"[INFO] 使用設備: {device_str}")
    set_random_seed(cfg.GENERAL.SEED)
    
    # 建立模型
    model = HRNetV2(
        C=cfg.MODEL.C,
        num_class=cfg.DATASET.NUM_CLASSES,
        in_ch=cfg.MODEL.IN_CHANNELS,
        use3D=(cfg.MODEL.USE_3D == True),
        useAttention=(cfg.MODEL.USE_ATTENTION == True)
    )
    
    # 載入檢查點
    ckpt = cfg.TEST.CHECKPOINT
    if ckpt and os.path.isfile(ckpt):
        print(f"[INFO] 從 {ckpt} 加載檢查點")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        raise ValueError(f"檢查點 {ckpt} 不存在")
    
    model.to(device)
    
    # 準備結果保存目錄
    results_dir = cfg.TEST.RESULTS_PATH
    os.makedirs(results_dir, exist_ok=True)
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 評估模式
    if args.mode == "original":
        # 僅評估原始數據集
        print("[INFO] 評估原始數據集...")
        
        # 載入原始數據集
        orig_dataset = forgeryHSIDataset(
            root=cfg.DATASET.ROOT,
            flist=cfg.DATASET.Val_data,
            split='val',
            target_type='mask',
            transform=DummyTransform()
        )
        orig_dataloader = DataLoader(
            orig_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            drop_last=False
        )
        
        # 評估
        results = evaluate_on_dataset(
            model, 
            orig_dataloader, 
            device, 
            cfg.DATASET.NUM_CLASSES,
            save_dir=vis_dir
        )
        
        print("[INFO] 原始數據集評估結果:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        
        # 保存結果
        results_path = os.path.join(results_dir, "original_evaluation.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"[INFO] 評估結果已保存到: {results_path}")
        
    elif args.mode == "adversarial":
        # 評估對抗攻擊效果
        print("[INFO] 評估對抗攻擊效果...")
        
        # 載入原始數據集
        orig_dataset = forgeryHSIDataset(
            root=cfg.DATASET.ROOT,
            flist=cfg.DATASET.Val_data,
            split='val',
            target_type='mask',
            transform=DummyTransform()
        )
        orig_dataloader = DataLoader(
            orig_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            drop_last=False
        )
        
        # 載入對抗數據集
        adv_dataset = forgeryHSIDataset(
            root=args.adv_dir,
            flist=args.adv_flist,
            split='adv',
            target_type='mask',
            transform=DummyTransform()
        )
        adv_dataloader = DataLoader(
            adv_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            drop_last=False
        )
        
        # 評估
        results = evaluate_adversarial(
            model, 
            orig_dataloader, 
            adv_dataloader, 
            device, 
            cfg.DATASET.NUM_CLASSES,
            save_dir=vis_dir
        )
        
        print("[INFO] 評估結果:")
        print("原始數據集:")
        for k, v in results["Original"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
                
        print("對抗數據集:")
        for k, v in results["Adversarial"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
                
        print("攻擊指標:")
        for k, v in results["Attack_Metrics"].items():
            print(f"  {k}: {v:.4f}")
        
        # 保存結果
        results_path = os.path.join(results_dir, "adversarial_evaluation.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"[INFO] 評估結果已保存到: {results_path}")
    
    else:
        raise ValueError(f"不支援的評估模式: {args.mode}")
    
    print("[INFO] 評估完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評估高光譜影像偽造檢測模型")
    parser.add_argument("--config_path", type=str, default="./config/train_config.yaml",
                      help="配置文件路徑")
    parser.add_argument("--mode", type=str, choices=["original", "adversarial"], default="original",
                      help="評估模式: 'original'僅評估原始數據集, 'adversarial'評估對抗攻擊效果")
    parser.add_argument("--adv_dir", type=str, default="",
                      help="對抗樣本目錄")
    parser.add_argument("--adv_flist", type=str, default="",
                      help="對抗樣本文件列表")
    args = parser.parse_args()
    main(args) 