import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
import time
import sys
from collections import OrderedDict
import random
import json
from datetime import datetime
import logging
import torch.cuda.amp as amp
import lpips
from torch_fidelity import calculate_metrics
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import get_cfg
from datasets.dataset import forgeryHSIDataset
from models.hrnet import HRNetV2
from utils.metrics import SegMetrics, AdvMetrics
from attacks.basic_attacks import fgsm_attack_adaptive, pgd_attack
from attacks.advanced_attacks import cw_attack, deepfool_attack
from attacks.hybrid_attacks import fgsm_hybrid_attack, pgd_hybrid_attack, cw_hybrid_attack
from attacks.spectral_attacks import fgsm_spectral_attack, pgd_spectral_attack, cw_spectral_attack, deepfool_spectral_attack
from attacks.spatial_attacks import fgsm_spatial_attack, pgd_spatial_attack, cw_spatial_attack, deepfool_spatial_attack
from utils.transforms import DummyTransform, PairCompose
from utils.utils import get_datetime_str, set_random_seed

# 導入可視化模塊
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.visualization import visualize_adversarial


def main(args):
    # 讀取配置
    cfg = get_cfg()
    if args.config_path:
        cfg.merge_from_file(args.config_path)
    cfg.freeze()

    # 設置設備
    device_ids = cfg.CUDA.CUDA_NUM
    device_str = f"cuda:{device_ids[0]}" if (cfg.CUDA.USE_CUDA and torch.cuda.is_available()) else 'cpu'
    device = torch.device(device_str)
    print(f"[INFO] 使用設備: {device_str}")
    set_random_seed(cfg.GENERAL.SEED)

    # 準備資料集
    test_dataset = forgeryHSIDataset(
        root=cfg.DATASET.ROOT,
        flist=cfg.DATASET.Val_data,
        split='val',
        target_type='mask',
        transform=DummyTransform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.ATTACK.BATCH_SIZE if cfg.ATTACK.BATCH_SIZE else 1,
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

    # 載入檢查點
    ckpt = cfg.TEST.CHECKPOINT
    if ckpt and os.path.isfile(ckpt):
        print(f"[INFO] 從 {ckpt} 加載檢查點")
        checkpoint = torch.load(ckpt, map_location=device)
        
        # 從檢查點中提取模型權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果沒有 model_state_dict 鍵，則假設整個檢查點就是模型權重
            model.load_state_dict(checkpoint)
            
        print("[INFO] 模型權重載入成功")
    else:
        raise ValueError(f"檢查點 {ckpt} 不存在")

    model.to(device)
    model.eval()

    # 設置攻擊參數
    attack_method = args.attack
    attack_domain = args.domain
    eps = cfg.ATTACK.EPS
    alpha = cfg.ATTACK.ALPHA if cfg.ATTACK.ALPHA else eps / 10
    steps = cfg.ATTACK.STEPS if cfg.ATTACK.STEPS else 10
    c = cfg.ATTACK.C if cfg.ATTACK.C else 0.01
    kappa = cfg.ATTACK.KAPPA if cfg.ATTACK.KAPPA else 0
    lr = cfg.ATTACK.LR if cfg.ATTACK.LR else 0.01
    max_iter = cfg.ATTACK.MAX_ITER if cfg.ATTACK.MAX_ITER else 50
    overshoot = cfg.ATTACK.OVERSHOOT if cfg.ATTACK.OVERSHOOT else 0.02
    target_bands = None  # 為光譜域攻擊指定的目標波段，如果為None則自動選擇

    # 設置時間戳和保存目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 設置保存目錄
    output_dir = cfg.ATTACK.SAVE_DIR
    save_dir = os.path.join(output_dir, f"{attack_domain}_{attack_method}_eps{eps}_{timestamp}")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    adv_dir = os.path.join(save_dir, "adversarial_samples")
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 設置日誌
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 設置日誌文件
    log_file = os.path.join(log_dir, f"attack_{attack_domain}_{attack_method}_eps{eps}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 記錄配置信息
    logging.info("=== 攻擊配置 ===")
    logging.info(f"攻擊方法: {attack_method}")
    logging.info(f"攻擊域: {attack_domain}")
    logging.info(f"擾動大小: {eps}")
    logging.info(f"批次大小: {cfg.ATTACK.BATCH_SIZE}")
    logging.info(f"設備: {device}")
    
    # 記錄數據集信息
    logging.info("\n=== 數據集信息 ===")
    logging.info(f"類別數: {cfg.DATASET.NUM_CLASSES}")
    
    # 記錄模型信息
    logging.info("\n=== 模型信息 ===")
    logging.info(f"通道數: {cfg.MODEL.C}")
    logging.info(f"輸入通道: {cfg.MODEL.IN_CHANNELS}")
    logging.info(f"使用3D卷積: {cfg.MODEL.USE_3D}")
    logging.info(f"使用注意力機制: {cfg.MODEL.USE_ATTENTION}")

    # 定義正規化參數（提前定義，以便在預檢查中使用）
    mean = cfg.DATASET.MEAN if hasattr(cfg.DATASET, 'MEAN') else 0.0
    std = cfg.DATASET.STD if hasattr(cfg.DATASET, 'STD') else 1.0
    
    # 初始化指標類
    seg_metrics_orig = SegMetrics(n_class=cfg.DATASET.NUM_CLASSES, device=device)
    seg_metrics_adv = SegMetrics(n_class=cfg.DATASET.NUM_CLASSES, device=device)
    adv_metrics = AdvMetrics(device=device)
    
    # 預檢查
    logging.info("\n=== 開始預檢查 ===")
    try:
        # 檢查模型加載
        with torch.no_grad():
            test_input = torch.randn(1, cfg.MODEL.IN_CHANNELS, 256, 256).to(device)
            test_output, _ = model(test_input)
            logging.info("模型加載檢查通過")
        
        # 檢查數據集
        test_batch = next(iter(test_loader))
        logging.info("數據集檢查通過")
        
        # 設置目錄（簡化檢查）
        os.makedirs(adv_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"預檢查失敗: {str(e)}")
        raise
    
    # 初始化統計數據
    start_batch = 0
    metrics = {
        'success_count': 0,
        'total_samples': 0,
        'l0_norms': [],
        'l2_norms': [],
        'linf_norms': [],
        'ssim_values': []
    }
    successful_samples = []
    
    # 定義全局變數來追蹤統計數據
    success_count = 0
    total_samples = 0
    l0_norms = []
    l2_norms = []
    linf_norms = []
    ssim_values = []
    
    # 選擇合適的損失函數
    criterion = nn.CrossEntropyLoss()
    
    # 選擇合適的攻擊函數
    if attack_domain == 'spatial':
        if attack_method == 'fgsm':
            attack_fn = fgsm_spatial_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_spatial_attack
        elif attack_method == 'cw':
            attack_fn = cw_spatial_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_spatial_attack
        else:
            raise ValueError(f"不支持的攻擊方法: {attack_method}")
    elif attack_domain == 'spectral':
        if attack_method == 'fgsm':
            attack_fn = fgsm_spectral_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_spectral_attack
        elif attack_method == 'cw':
            attack_fn = cw_spectral_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_spectral_attack
        else:
            raise ValueError(f"不支持的攻擊方法: {attack_method}")
    elif attack_domain == 'hybrid':
        if attack_method == 'fgsm':
            attack_fn = fgsm_hybrid_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_hybrid_attack
        elif attack_method == 'cw':
            attack_fn = cw_hybrid_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_spatial_attack  # 暫時使用空間版本
        else:
            raise ValueError(f"不支持的攻擊方法: {attack_method}")
    else:
        if attack_method == 'fgsm':
            attack_fn = fgsm_attack_adaptive
        elif attack_method == 'pgd':
            attack_fn = pgd_attack
        elif attack_method == 'cw':
            attack_fn = cw_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_attack
        else:
            raise ValueError(f"不支持的攻擊方法: {attack_method}")
    
    # 開始攻擊
    logging.info("\n=== 開始攻擊 ===")
    pbar = tqdm(test_loader, desc="生成對抗樣本")
    for batch_idx, (images, labels, filenames) in enumerate(pbar):
        try:
            images = images.to(device)
            # 確保標籤是Long類型
            labels = labels.to(device).long()
            
            # 確保 filenames 是字符串列表
            if isinstance(filenames, torch.Tensor):
                filenames = [f"sample_{batch_idx}_{i}" for i in range(len(filenames))]
            
            # 進行原始預測
            outputs_orig, _ = model(images)
            preds_orig = outputs_orig.argmax(dim=1)
            
            # 更新原始圖像的分割指標
            seg_metrics_orig.update(labels, preds_orig)
            
            # 生成對抗樣本
            if attack_domain == 'hybrid':
                # 混合攻擊特有參數
                spatial_weight = cfg.ATTACK.HYBRID.SPATIAL_WEIGHT
                target_bands = cfg.ATTACK.HYBRID.TARGET_BANDS
                
                # 生成對抗樣本
                if attack_method == 'fgsm':
                    adv_images = attack_fn(
                        model, images, labels, eps, criterion, device, mean, std,
                        spatial_weight=spatial_weight, target_bands=target_bands
                    )
                elif attack_method == 'pgd':
                    adv_images = attack_fn(
                        model, images, labels, eps, alpha, steps, criterion, device, mean, std,
                        spatial_weight=spatial_weight, target_bands=target_bands
                    )
                elif attack_method == 'cw':
                    adv_images = attack_fn(
                        model, images, labels, c=cfg.ATTACK.C, kappa=cfg.ATTACK.KAPPA,
                        steps=cfg.ATTACK.STEPS, lr=cfg.ATTACK.LR, eps=eps,
                        device=device, mean=mean, std=std,
                        spatial_weight=spatial_weight, target_bands=target_bands
                    )
                elif attack_method == 'deepfool':
                    adv_images = attack_fn(
                        model, images, num_classes=cfg.DATASET.NUM_CLASSES,
                        max_iter=cfg.ATTACK.MAX_ITER, overshoot=cfg.ATTACK.OVERSHOOT,
                        device=device, mean=mean, std=std,
                        spatial_weight=spatial_weight, target_bands=target_bands
                    )
            elif attack_domain in ['spatial', 'spectral']:
                if attack_method == 'fgsm':
                    adv_images = attack_fn(
                        model, images, labels, eps, criterion, device, mean, std
                    )
                elif attack_method == 'pgd':
                    adv_images = attack_fn(
                        model, images, labels, eps, alpha, steps, criterion, device, mean, std
                    )
                elif attack_method == 'cw':
                    adv_images = attack_fn(
                        model, images, labels, c=cfg.ATTACK.C, kappa=cfg.ATTACK.KAPPA,
                        steps=cfg.ATTACK.STEPS, lr=cfg.ATTACK.LR, eps=eps,
                        device=device, mean=mean, std=std
                    )
                elif attack_method == 'deepfool':
                    adv_images = attack_fn(
                        model, images, num_classes=cfg.DATASET.NUM_CLASSES,
                        max_iter=cfg.ATTACK.MAX_ITER, overshoot=cfg.ATTACK.OVERSHOOT,
                        device=device, mean=mean, std=std
                    )
            else:
                raise ValueError(f"不支持的攻擊域: {attack_domain}")
            
            # 對抗樣本的預測
            outputs_adv, _ = model(adv_images)
            preds_adv = outputs_adv.argmax(dim=1)
            
            # 更新對抗樣本的分割指標
            seg_metrics_adv.update(labels, preds_adv)
            
            # 更新對抗攻擊指標
            adv_metrics.update(preds_orig, preds_adv, images, adv_images, labels)
            
            # 計算擾動的範數
            perturbation = adv_images - images
            
            # 保存樣本和計算指標
            for i in range(images.size(0)):
                total_samples += 1
                
                # 計算擾動範數
                p = perturbation[i].cpu().numpy()
                l0_norm = np.count_nonzero(p) / p.size
                l2_norm = np.sqrt(np.sum(p**2))
                linf_norm = np.max(np.abs(p))
                
                l0_norms.append(l0_norm)
                l2_norms.append(l2_norm)
                linf_norms.append(linf_norm)
                
                # 計算SSIM
                ssim = 1.0 - (l2_norm / (l2_norm + 1.0))
                ssim_values.append(ssim)
                
                # 檢查攻擊是否成功
                is_success = (preds_orig[i] != preds_adv[i]).any().item()
                
                if is_success:
                    success_count += 1
                    successful_samples.append(filenames[i])
                    
                    # 可視化成功的對抗樣本
                    vis_path = os.path.join(vis_dir, f"{filenames[i]}_adv.png")
                    visualize_adversarial(images[i], adv_images[i], labels[i], preds_orig[i], preds_adv[i], vis_path, 'test', args)
                
                # 保存對抗樣本
                base_name = filenames[i]
                adv_path = os.path.join(adv_dir, f"{base_name}_adv.npy")
                np.save(adv_path, adv_images[i].cpu().numpy())

                # 更新進度條
                current_success_rate = success_count/total_samples if total_samples > 0 else 0
                pbar.set_postfix({
                    "成功率": f"{current_success_rate:.2%}",
                    "處理樣本": f"{total_samples}"
                })
        
        except Exception as e:
            logging.error(f"批次 {batch_idx} 處理失敗: {str(e)}")
            # 更新metrics字典
            metrics = {
                'success_count': success_count,
                'total_samples': total_samples,
                'l0_norms': l0_norms,
                'l2_norms': l2_norms,
                'linf_norms': linf_norms,
                'ssim_values': ssim_values
            }
            raise
    
    # 更新metrics字典，以便後續使用
    metrics = {
        'success_count': success_count,
        'total_samples': total_samples,
        'l0_norms': l0_norms,
        'l2_norms': l2_norms,
        'linf_norms': linf_norms,
        'ssim_values': ssim_values
    }
    
    # 計算最終結果
    try:
        # 計算原始性能
        orig_results = seg_metrics_orig.get_results()
        if not orig_results:
            orig_results = {
                'Overall acc': 0.0,
                'Mean acc': 0.0,
                'Mean IoU': 0.0,
                'Class IoU': {}
            }
        
        # 確保所有值都是 JSON 可序列化的
        for key in orig_results:
            if isinstance(orig_results[key], np.ndarray) or isinstance(orig_results[key], np.number):
                orig_results[key] = float(orig_results[key])
        
        # 計算對抗性能
        adv_results = seg_metrics_adv.get_results()
        if not adv_results:
            adv_results = {
                'Overall acc': 0.0,
                'Mean acc': 0.0,
                'Mean IoU': 0.0,
                'Class IoU': {}
            }
        
        # 確保所有值都是 JSON 可序列化的
        for key in adv_results:
            if isinstance(adv_results[key], np.ndarray) or isinstance(adv_results[key], np.number):
                adv_results[key] = float(adv_results[key])
        
        # 計算攻擊指標
        attack_results = adv_metrics.get_results()
        if not attack_results:
            attack_results = {
                'Attack_Success_Rate': 0.0,
                'Significant_Attack_Success_Rate': 0.0,
                'Average_L2_Norm': 0.0,
                'Average_SSIM': 0.0
            }
        
        # 確保所有值都是 JSON 可序列化的
        for key in attack_results:
            if isinstance(attack_results[key], np.ndarray) or isinstance(attack_results[key], np.number):
                attack_results[key] = float(attack_results[key])
        
        # 計算性能下降 - 修復Performance_Drop計算
        performance_drop = {
            'Pixel_Accuracy_Drop': float(orig_results.get('Overall acc', 0.0)) - float(adv_results.get('Overall acc', 0.0)),
            'Mean_Accuracy_Drop': float(orig_results.get('Mean acc', 0.0)) - float(adv_results.get('Mean acc', 0.0)),
            'Mean_IoU_Drop': float(orig_results.get('Mean IoU', 0.0)) - float(adv_results.get('Mean IoU', 0.0)),
        }
        
        # 計算各類別IoU的下降
        classes_iou_drop = {}
        orig_class_iou = orig_results.get('Class IoU', {})
        adv_class_iou = adv_results.get('Class IoU', {})
        
        # 合併所有類別的鍵
        all_classes = set(orig_class_iou.keys()).union(set(adv_class_iou.keys()))
        for cls in all_classes:
            orig_iou = float(orig_class_iou.get(cls, 0.0))
            adv_iou = float(adv_class_iou.get(cls, 0.0))
            classes_iou_drop[cls] = orig_iou - adv_iou
        
        performance_drop['Class_IoU_Drop'] = classes_iou_drop
        
        # 計算F1分數
        f1_scores_orig = {}
        f1_scores_adv = {}
        
        for cls in all_classes:
            # 原始F1分數: 2 * precision * recall / (precision + recall)
            # IoU = TP / (TP + FP + FN)
            # F1 = 2 * TP / (2 * TP + FP + FN)
            # 從IoU轉換到F1: F1 = 2 * IoU / (1 + IoU)
            orig_iou = float(orig_class_iou.get(cls, 0.0))
            adv_iou = float(adv_class_iou.get(cls, 0.0))
            
            if orig_iou > 0:
                f1_scores_orig[cls] = 2 * orig_iou / (1 + orig_iou)
            else:
                f1_scores_orig[cls] = 0.0
                
            if adv_iou > 0:
                f1_scores_adv[cls] = 2 * adv_iou / (1 + adv_iou)
            else:
                f1_scores_adv[cls] = 0.0
        
        # 計算F1分數下降
        f1_drop = {}
        for cls in all_classes:
            f1_drop[cls] = f1_scores_orig.get(cls, 0.0) - f1_scores_adv.get(cls, 0.0)
        
        # 計算平均F1分數
        mean_f1_orig = sum(f1_scores_orig.values()) / len(f1_scores_orig) if f1_scores_orig else 0.0
        mean_f1_adv = sum(f1_scores_adv.values()) / len(f1_scores_adv) if f1_scores_adv else 0.0
        
        performance_drop['Mean_F1_Drop'] = mean_f1_orig - mean_f1_adv
        performance_drop['Class_F1_Drop'] = f1_drop
        
        # 添加成功率解釋
        attack_explanation = {
            'Standard_Success_Definition': '任何像素預測改變',
            'Significant_Success_Definition': f'至少{adv_metrics.significant_pixel_ratio:.1%}的像素預測改變',
            'Class_Success_Definition': '指定類別中有像素預測改變',
        }
        
        # 保存最終結果
        metrics_report = {
            'Original_Performance': orig_results,
            'Adversarial_Performance': adv_results,
            'Performance_Drop': performance_drop,
            'Attack_Metrics': attack_results,
            'Attack_Definitions': attack_explanation
        }
        
        # 確保 JSON 序列化安全
        with open(os.path.join(save_dir, 'metrics_report.json'), 'w') as f:
            json.dump(metrics_report, f, indent=4, cls=NumpyEncoder)
        
        # 記錄最終結果
        logging.info("\n=== 攻擊完成 ===")
        logging.info(f"總樣本數: {metrics['total_samples']}")
        logging.info(f"成功樣本數: {metrics['success_count']}")
        logging.info(f"最終成功率: {metrics['success_count']/metrics['total_samples'] if metrics['total_samples'] > 0 else 0:.2%}")
        logging.info(f"平均L2範數: {float(np.mean(metrics['l2_norms']) if metrics['l2_norms'] else 0):.4f}")
        logging.info(f"平均SSIM: {float(np.mean(metrics['ssim_values']) if metrics['ssim_values'] else 0):.4f}")
        
    except Exception as e:
        logging.error(f"計算最終結果時出錯: {str(e)}")
        raise

    # 計算並保存攻擊結果
    success_rate = success_count / total_samples if total_samples > 0 else 0
    avg_l0_norm = float(np.mean(l0_norms) if l0_norms else 0)
    avg_l2_norm = float(np.mean(l2_norms) if l2_norms else 0)
    avg_linf_norm = float(np.mean(linf_norms) if linf_norms else 0)
    avg_ssim = float(np.mean(ssim_values) if ssim_values else 0)

    results = {
        "attack_method": attack_method,
        "attack_domain": attack_domain,
        "eps": float(eps),
        "success_rate": float(success_rate),
        "total_samples": int(total_samples),
        "success_count": int(success_count),
        "avg_l0_norm": float(avg_l0_norm),
        "avg_l2_norm": float(avg_l2_norm),
        "avg_linf_norm": float(avg_linf_norm),
        "avg_ssim": float(avg_ssim)
    }

    results_path = os.path.join(save_dir, "attack_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # 保存成功樣本的文件名
    successful_samples_path = os.path.join(save_dir, "successful_samples.txt")
    with open(successful_samples_path, "w") as f:
        for filename in successful_samples:
            f.write(f"{os.path.basename(filename)}\n")

    # 創建filelist以便後續評估
    filelist_path = os.path.join(save_dir, "adv_filelist.txt")
    with open(filelist_path, "w") as f:
        for _, _, files in os.walk(adv_dir):
            for file in files:
                if file.endswith(".npy"):
                    f.write(f"{file}\n")

    print(f"[INFO] 攻擊完成，結果保存在 {save_dir}")
    print(f"[INFO] 攻擊成功率: {success_rate:.4f} ({success_count}/{total_samples})")
    print(f"[INFO] 平均L2範數: {avg_l2_norm:.4f}")
    print(f"[INFO] 平均L0範數: {avg_l0_norm:.4f}")
    print(f"[INFO] 平均L∞範數: {avg_linf_norm:.4f}")
    print(f"[INFO] 平均SSIM: {avg_ssim:.4f}")

    # 可視化對抗樣本
    if args.n_visualize > 0:
        if args.verbose:
            print(f"\n生成 {min(args.n_visualize, len(images))} 個樣本的可視化結果...")
        
        for i in range(min(args.n_visualize, len(images))):
            # 獲取文件名作為可視化文件名
            base_name = os.path.basename(paths[i]).split('.')[0]
            vis_path = os.path.join(vis_dir, f"{base_name}_adv.png")
            
            # 使用改進的可視化函數
            visualize_adversarial(
                orig_image=images[i], 
                adv_image=adv_images[i], 
                label=labels[i], 
                pred_orig=preds_orig[i], 
                pred_adv=preds_adv[i], 
                output_path=vis_path, 
                mode='test', 
                args=args
            )


# 添加一個 JSON 編碼器來處理 NumPy 數據類型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成對抗樣本")
    parser.add_argument("--config_path", type=str, default="./config/attack_config.yaml", help="配置文件路徑")
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw', 'deepfool'], help='Attack method')
    parser.add_argument('--domain', type=str, default='spatial', choices=['spatial', 'spectral', 'hybrid'], help='Attack domain')
    parser.add_argument('--n_visualize', type=int, default=0, help='Number of adversarial samples to visualize')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    main(args) 