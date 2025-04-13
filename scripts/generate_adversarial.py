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
import pickle
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import get_cfg
from datasets.dataset import forgeryHSIDataset
from models.hrnet import HRNetV2
from utils.metrics import SegMetrics, AdvMetrics
from attacks.basic_attacks import fgsm_attack_adaptive, pgd_attack
from attacks.hybrid_attacks import fgsm_hybrid_attack, pgd_hybrid_attack, cw_hybrid_attack
from attacks.spectral_attacks import fgsm_spectral_attack, pgd_spectral_attack, cw_spectral_attack, deepfool_spectral_attack
from attacks.spatial_attacks import fgsm_spatial_attack, pgd_spatial_attack, cw_spatial_attack, deepfool_spatial_attack
from utils.transforms import DummyTransform, PairCompose
from utils.utils import get_datetime_str, set_random_seed, get_dataloader
from utils.image_utils import save_image, save_rgb, save_rgb_comparison, visualize_attack_comparison
from attacks.utils import normalize, unnormalize

# 導入可視化模塊
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.visualization import visualize_adversarial


def main():
    print("對抗樣本生成")
    parser = argparse.ArgumentParser(description='生成對抗樣本')
    
    # 基本配置
    parser.add_argument('--config', type=str, default='config/attack_config.yaml', help='配置檔案路徑')
    parser.add_argument('--model_path', type=str, default='checkpoints_dfcn_improved/best.pth', help='目標模型路徑')
    parser.add_argument('--model_type', type=str, default='hrnet', help='模型類型')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入程序數')
    parser.add_argument('--device', type=str, default='cuda', help='使用設備')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU編號')
    parser.add_argument('--save_dir', type=str, default='results/adversarial', help='結果保存目錄')
    
    # 攻擊配置
    parser.add_argument('--attack_method', type=str, default='pgd', 
                       help='攻擊方法: fgsm, pgd, deepfool, cw')
    parser.add_argument('--attack_domain', type=str, default='spatial',
                       help='攻擊域: spatial, spectral, hybrid')
    parser.add_argument('--norm', type=str, default='linf', help='攻擊範數: l2, linf')
    parser.add_argument('--eps', type=float, default=0.03, help='最大擾動大小')
    parser.add_argument('--alpha', type=float, default=0.01, help='PGD步長')
    parser.add_argument('--steps', type=int, default=40, help='PGD/CW迭代步數')
    parser.add_argument('--random_start', action='store_true', default=True, help='是否使用隨機初始化')
    parser.add_argument('--num_classes', type=int, default=2, help='分類數量')
    
    # 混合攻擊配置
    parser.add_argument('--spatial_weight', type=float, default=0.5, help='混合攻擊中空間權重, 0-1, 僅用於hybrid類攻擊')
    parser.add_argument('--target_bands', type=str, default=None, 
                       help='要攻擊的特定波段，以逗號分隔，例如"0,10,20"，僅用於spectral和hybrid攻擊')
    
    # 感知約束配置
    parser.add_argument('--perceptual_constraint', action='store_true', default=True, help='是否應用感知約束')
    parser.add_argument('--ssim_threshold', type=float, default=0.95, help='SSIM閾值，高於此值視為可接受')
    parser.add_argument('--lpips_threshold', type=float, default=0.05, help='LPIPS閾值，低於此值視為可接受')
    
    # 自適應擾動大小配置
    parser.add_argument('--adaptive_eps', action='store_true', default=True, help='是否使用自適應擾動大小')
    parser.add_argument('--min_eps', type=float, default=0.01, help='最小擾動大小')
    parser.add_argument('--max_eps', type=float, default=0.1, help='最大擾動大小')
    
    # 光譜重要性分析配置
    parser.add_argument('--spectral_importance', action='store_true', default=True, help='是否使用光譜重要性分析')
    parser.add_argument('--spectral_threshold', type=float, default=0.7, help='光譜重要性閾值，累積重要性高於此值的波段會被選中')
    
    # 評估配置
    parser.add_argument('--samples', type=int, default=5, help='生成樣本數量')
    parser.add_argument('--visualize', action='store_true', default=True, help='是否可視化結果')
    parser.add_argument('--save_images', action='store_true', default=True, help='是否保存圖像')
    parser.add_argument('--save_perturbations', action='store_true', default=True, help='是否保存擾動')
    parser.add_argument('--save_metrics', action='store_true', default=True, help='是否保存指標')
    
    args = parser.parse_args()
    
    # 確保保存目錄存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 載入配置
    config = get_cfg()
    # 如果指定了配置文件，從YAML加載配置
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # 對於嵌套字典，需要遞歸設置
                    for sub_key, sub_value in value.items():
                        if hasattr(config, key):
                            if isinstance(sub_value, dict):
                                # 如果子值仍然是字典，將其轉換為基本類型
                                for k, v in sub_value.items():
                                    if hasattr(getattr(config, key), sub_key):
                                        setattr(getattr(getattr(config, key), sub_key), k, v)
                            else:
                                setattr(getattr(config, key), sub_key, sub_value)
                else:
                    if hasattr(config, key):
                        setattr(config, key, value)
    
    # 設定設備
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        # 設置當前設備，某些操作會默認使用這個設備
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device('cpu')
    print(f"使用設備: {device} (GPU {args.gpu_id if args.device == 'cuda' and torch.cuda.is_available() else 'None'})")
    
    # 從配置文件中讀取GPU設置（如果命令行未指定）
    if hasattr(config, 'CUDA') and hasattr(config.CUDA, 'CUDA_NUM') and config.CUDA.CUDA_NUM and args.gpu_id == 0:
        # 僅當命令行參數使用默認值時，才使用配置文件中的設置
        cuda_num = config.CUDA.CUDA_NUM
        if isinstance(cuda_num, list) and len(cuda_num) > 0:
            args.gpu_id = cuda_num[0]
            if args.device == 'cuda' and torch.cuda.is_available():
                device = torch.device(f'cuda:{args.gpu_id}')
                torch.cuda.set_device(args.gpu_id)
                print(f"從配置文件中讀取GPU設置，更新設備: {device} (GPU {args.gpu_id})")
    
    # 獲取項目根目錄
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 實際數據路徑
    real_data_root = '/ssd8/HyperForensics_Data/HyperForensics_Dataset'
    
    # 載入資料集
    try:
        print("嘗試加載資料集...")
        # 使用實際數據路徑
        print(f"使用實際數據路徑: {real_data_root}")
        val_loader = get_dataloader(args.batch_size, args.num_workers, root_dir=real_data_root, split='val')
    except FileNotFoundError as e:
        print(f"警告: {str(e)}")
        print("嘗試使用替代方法載入資料集...")
        from datasets.dataset import forgeryHSIDataset
        
        # 使用絕對路徑讀取文件
        flist_path = os.path.join(project_root, 'val_all.txt')
        if not os.path.exists(flist_path):
            raise FileNotFoundError(f"找不到資料集文件: {flist_path}")

        # 測試用建立數據集
        test_dataset = forgeryHSIDataset(
            root='data',
            flist=flist_path,
            split='val',
            target_type='mask',
            transform=None
        )
        
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
    print(f"資料集已載入，批次大小: {args.batch_size}")
    
    # 載入模型
    if args.model_type.lower() == 'hrnet':
        from models.hrnet import HRNetV2
        model = HRNetV2(
            C=config.MODEL.C,
            num_class=config.DATASET.NUM_CLASSES,
            in_ch=config.MODEL.IN_CHANNELS,
            use3D=(config.MODEL.USE_3D == True),
            useAttention=(config.MODEL.USE_ATTENTION == True)
        )
    elif args.model_type.lower() == 'deeplabv3':
        from models.deeplabv3 import DeepLabV3
        model = DeepLabV3(config)
    else:
        raise ValueError(f"不支持的模型類型: {args.model_type}")
    
    # 載入權重
    print(f"從 {args.model_path} 加載檢查點")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 從檢查點中提取模型權重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # 如果沒有特定的鍵，則假設整個檢查點就是模型權重
        model.load_state_dict(checkpoint)
    
    print("模型權重載入成功")
    
    model = model.to(device)
    model.eval()
    print(f"模型已載入: {args.model_path}")
    
    # 設定攻擊函數
    attack_method = args.attack_method.lower()
    attack_domain = args.attack_domain.lower()
    attack_fn = None
    
    # 解析目標波段
    target_bands = None
    if args.target_bands:
        target_bands = [int(b) for b in args.target_bands.split(',')]
        print(f"目標波段: {target_bands}")
    
    # 選擇攻擊函數
    if attack_method == 'fgsm':
        if attack_domain == 'spatial':
            attack_fn = fgsm_spatial_attack
        elif attack_domain == 'spectral':
            attack_fn = fgsm_spectral_attack
        elif attack_domain == 'hybrid':
            attack_fn = fgsm_hybrid_attack
        else:
            attack_fn = fgsm_attack_adaptive
    elif attack_method == 'pgd':
        if attack_domain == 'spatial':
            attack_fn = pgd_spatial_attack
        elif attack_domain == 'spectral':
            attack_fn = pgd_spectral_attack
        elif attack_domain == 'hybrid':
            attack_fn = pgd_hybrid_attack
        else:
            attack_fn = pgd_attack
    elif attack_method == 'deepfool':
        if attack_domain == 'spatial':
            attack_fn = deepfool_spatial_attack
        elif attack_domain == 'hybrid':
            attack_fn = deepfool_hybrid_attack
        else:
            attack_fn = deepfool_spectral_attack
    elif attack_method == 'cw':
        if attack_domain == 'spatial':
            attack_fn = cw_spatial_attack
        elif attack_domain == 'spectral':
            attack_fn = cw_spectral_attack
        elif attack_domain == 'hybrid':
            attack_fn = cw_hybrid_attack
        else:
            attack_fn = cw_attack
    else:
        raise ValueError(f"不支持的攻擊方法: {attack_method}")
    
    print(f"使用攻擊方法: {attack_method}, 攻擊域: {attack_domain}")
    
    # 创建攻擊設定字典
    attack_kwargs = {
        'apply_perceptual_constraint': args.perceptual_constraint,
        'ssim_threshold': args.ssim_threshold,
        'lpips_threshold': args.lpips_threshold,
        'use_adaptive_eps': args.adaptive_eps,
        'min_eps': args.min_eps,
        'max_eps': args.max_eps,
        'use_spectral_importance': args.spectral_importance,
        'spectral_threshold': args.spectral_threshold
    }
    
    if attack_domain in ['spectral', 'hybrid'] and target_bands is not None:
        attack_kwargs['target_bands'] = target_bands
        
    if attack_domain == 'hybrid':
        attack_kwargs['spatial_weight'] = args.spatial_weight
        
    # 設定指標計算器
    adv_metrics = AdvMetrics(device=device, significant_pixel_ratio=0.05)
    
    # 資料統計
    clean_preds = []
    adv_preds = []
    clean_images = []
    adv_images = []
    perturbations = []
    labels_list = []
    
    # 攻擊資料
    loader = tqdm(val_loader, desc="生成對抗樣本")
    sample_count = 0
    
    for i, data in enumerate(loader):
        if sample_count >= args.samples:
            break
            
        # 處理不同長度的返回值
        if isinstance(data, tuple) or isinstance(data, list):
            if len(data) == 3:
                images, labels, _ = data  # 如果返回三個值，忽略第三個
            elif len(data) == 2:
                images, labels = data
            else:
                raise ValueError(f"意外的數據格式，長度為 {len(data)}")
        else:
            raise ValueError(f"意外的數據格式: {type(data)}")
        
        images = images.to(device)
        labels = labels.to(device)
        
        # 使用交叉熵損失
        criterion = torch.nn.CrossEntropyLoss()
        
        # 計算乾淨預測
        with torch.no_grad():
            clean_output = model(images)
            if isinstance(clean_output, tuple):
                clean_output = clean_output[0]
            clean_pred = clean_output.argmax(1)
        
        # 生成對抗樣本
        if attack_method == 'fgsm':
            adv_images_batch = attack_fn(
                model, images, labels, args.eps, criterion, device, 
                config.DATASET.MEAN, config.DATASET.STD, **attack_kwargs
            )
        elif attack_method == 'pgd':
            adv_images_batch = attack_fn(
                model, images, labels, args.eps, args.alpha, args.steps, 
                criterion, device, config.DATASET.MEAN, config.DATASET.STD, **attack_kwargs
            )
        elif attack_method == 'deepfool':
            adv_images_batch = attack_fn(
                model, images, args.num_classes, args.steps, 0.02, 
                device, config.DATASET.MEAN, config.DATASET.STD, **attack_kwargs
            )
        elif attack_method == 'cw':
            adv_images_batch = attack_fn(
                model, images, labels, 0.01, 0, args.steps, 0.01, args.eps,
                device, config.DATASET.MEAN, config.DATASET.STD, **attack_kwargs
            )
            
        # 計算對抗預測
        with torch.no_grad():
            adv_output = model(adv_images_batch)
            if isinstance(adv_output, tuple):
                adv_output = adv_output[0]
            adv_pred = adv_output.argmax(1)
        
        # 計算擾動
        pert = adv_images_batch - images
        
        # 更新指標
        batch_metrics = {
            'attack_method': attack_method,
            'attack_domain': attack_domain,
            'eps': args.eps,
            'perceptual_constraint': args.perceptual_constraint,
            'ssim_threshold': args.ssim_threshold,
            'lpips_threshold': args.lpips_threshold,
            'adaptive_eps': args.adaptive_eps,
            'min_eps': args.min_eps,
            'max_eps': args.max_eps,
            'spectral_importance': args.spectral_importance,
            'spectral_threshold': args.spectral_threshold
        }
        
        if attack_domain == 'hybrid':
            batch_metrics['spatial_weight'] = args.spatial_weight
            
        adv_metrics.update(images, adv_images_batch, labels, clean_pred, adv_pred, batch_metrics)
        
        # 保存結果
        clean_preds.append(clean_pred.cpu())
        adv_preds.append(adv_pred.cpu())
        clean_images.append(images.cpu())
        adv_images.append(adv_images_batch.cpu())
        perturbations.append(pert.cpu())
        labels_list.append(labels.cpu())
        
        sample_count += images.size(0)
        loader.set_description(f"已生成 {sample_count}/{args.samples} 樣本")
    
    # 合併批次
    clean_preds = torch.cat(clean_preds, dim=0)
    adv_preds = torch.cat(adv_preds, dim=0)
    clean_images = torch.cat(clean_images, dim=0)
    adv_images = torch.cat(adv_images, dim=0)
    perturbations = torch.cat(perturbations, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    
    # 生成結果路徑
    attack_name = f"{attack_method}_{attack_domain}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.save_dir, f"{attack_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 創建子目錄
    images_dir = os.path.join(result_dir, "images")
    perturbations_dir = os.path.join(result_dir, "perturbations")
    metrics_dir = os.path.join(result_dir, "metrics")
    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(perturbations_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 設置日誌文件
    log_file = os.path.join(log_dir, f"{attack_name}.log")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    # 記錄攻擊配置
    logging.info("=== 攻擊配置 ===")
    logging.info(f"攻擊方法: {attack_method}")
    logging.info(f"攻擊域: {attack_domain}")
    logging.info(f"擾動大小: {args.eps}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"迭代步數: {args.steps}")
    logging.info(f"設備: {device}")
    
    # 保存指標
    if args.save_metrics:
        metrics_results = adv_metrics.get_results()
        
        # 計算攻擊成功率與指標
        success_rate = metrics_results['Attack_Success_Rate']
        significant_success_rate = metrics_results['Significant_Attack_Success_Rate']
        avg_l2_norm = metrics_results['Average_L2_Norm']
        avg_linf_norm = metrics_results['Average_Linf_Norm']
        
        # 計算SSIM和LPIPS
        avg_ssim = metrics_results.get('Average_SSIM', 0.0)
        avg_lpips = metrics_results.get('Average_LPIPS', 0.0)
        
        # 計算類別成功率
        class_success_rates = metrics_results.get('Class_Attack_Success_Rate', {})
        
        # 保存詳細指標報告 (JSON)
        metrics_file = os.path.join(metrics_dir, "metrics_report.json")
        
        # 創建簡化的指標報告
        report = {
            "攻擊配置": {
                "攻擊方法": attack_method,
                "攻擊域": attack_domain,
                "擾動大小": float(args.eps),
                "批次大小": args.batch_size,
                "迭代步數": args.steps
            },
            "攻擊指標": {
                "樣本數量": int(adv_metrics.total_samples),
                "成功樣本數": int(adv_metrics.success_count),
                "攻擊成功率": float(success_rate),
                "平均L2範數": float(avg_l2_norm),
                "平均L∞範數": float(avg_linf_norm),
                "平均SSIM": float(avg_ssim),
                "平均LPIPS": float(avg_lpips)
            },
            "效能變化": metrics_results.get("Performance_Changes", {}),
            "分割評估指標": metrics_results.get("Segmentation_Metrics", {})
        }
        
        # 添加偽造區域準確率到報表中
        if "Segmentation_Metrics" in metrics_results and "Accuracy_Details" in metrics_results["Segmentation_Metrics"]:
            acc_details = metrics_results["Segmentation_Metrics"]["Accuracy_Details"]
            if "original" in acc_details and "adversarial" in acc_details:
                # 獲取類別1 (偽造區域) 的準確率
                orig_forgery_acc = acc_details["original"].get("class_1", 0.0)
                adv_forgery_acc = acc_details["adversarial"].get("class_1", 0.0)
                
                # 計算變化率
                if orig_forgery_acc > 0:
                    change_rate = (adv_forgery_acc - orig_forgery_acc) / orig_forgery_acc
                else:
                    change_rate = 0.0
                
                # 添加到報表
                report["分割評估指標"]["偽造區域準確率"] = {
                    "原始": float(orig_forgery_acc),
                    "對抗": float(adv_forgery_acc),
                    "變化率": float(change_rate)
                }
        
        # 保存為JSON格式
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        logging.info(f"指標報告已保存到: {metrics_file}")
        
        # 保存簡化的摘要報告
        summary_file = os.path.join(result_dir, "summary.json")
        summary_data = {
            "攻擊方法": attack_method,
            "攻擊域": attack_domain,
            "擾動大小": float(args.eps),
            "攻擊成功率": float(success_rate),
            "平均L2範數": float(avg_l2_norm),
            "平均L∞範數": float(avg_linf_norm),
            "平均SSIM": float(avg_ssim),
            "平均LPIPS": float(avg_lpips),
            "樣本數量": int(adv_metrics.total_samples),
            "成功樣本數": int(adv_metrics.success_count)
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        logging.info(f"摘要報告已保存到: {summary_file}")
        
        # 列印關鍵指標
        print(f"攻擊成功率(ASR): {success_rate:.4f} ({adv_metrics.success_count}/{adv_metrics.total_samples})")
        print(f"平均L2範數: {avg_l2_norm:.4f}")
        print(f"平均L∞範數: {avg_linf_norm:.4f}")
        if 'Average_SSIM' in metrics_results:
            print(f"平均SSIM: {avg_ssim:.4f}")
        if 'Average_LPIPS' in metrics_results:
            print(f"平均LPIPS: {avg_lpips:.4f}")
            
        # 顯示效能變化指標
        if 'Performance_Changes' in metrics_results:
            perf_changes = metrics_results['Performance_Changes']
            
            # 顯示光譜影響
            if 'Spectral_Impact' in perf_changes:
                spectral_impact = perf_changes['Spectral_Impact']
                print(f"平均最大波段差異: {spectral_impact.get('avg_max_band_difference', 0.0):.4f}")
                top_bands = spectral_impact.get('avg_top_affected_bands', [])
                if top_bands:
                    print(f"最常受影響的波段: {', '.join(map(str, top_bands[:5]))}")
                    
            # 顯示預測變化
            if 'Prediction_Changes' in perf_changes and perf_changes['Prediction_Changes']:
                pred_changes = perf_changes['Prediction_Changes']
                # 找出最顯著的前三個預測變化
                sorted_changes = sorted(pred_changes.items(), key=lambda x: x[1], reverse=True)[:3]
                print("最顯著的預測變化:")
                for change, ratio in sorted_changes:
                    print(f"  {change}: {ratio*100:.2f}%")
            
        # 顯示分割評估指標
        if 'Segmentation_Metrics' in metrics_results and metrics_results['Segmentation_Metrics']:
            seg_metrics = metrics_results['Segmentation_Metrics']
            print("\n分割評估指標:")
            
            # 顯示mIoU
            if 'Mean_IoU' in seg_metrics:
                miou = seg_metrics['Mean_IoU']
                orig_miou = miou.get('original', 0.0)
                adv_miou = miou.get('adversarial', 0.0)
                change = miou.get('relative_change', 0.0) * 100
                print(f"  mIoU: {orig_miou:.4f} → {adv_miou:.4f} (變化率: {change:.2f}%)")
            
            # 顯示像素準確率
            if 'Pixel_Accuracy' in seg_metrics:
                acc = seg_metrics['Pixel_Accuracy']
                orig_acc = acc.get('original', 0.0)
                adv_acc = acc.get('adversarial', 0.0)
                change = acc.get('relative_change', 0.0) * 100
                print(f"  像素準確率: {orig_acc:.4f} → {adv_acc:.4f} (變化率: {change:.2f}%)")
                
            # 添加：顯示偽造區域準確率（類別1的準確率）
            if 'Accuracy_Details' in seg_metrics:
                acc_details = seg_metrics['Accuracy_Details']
                if 'original' in acc_details and 'adversarial' in acc_details:
                    # 獲取類別1 (偽造區域) 的準確率
                    orig_forgery_acc = acc_details['original'].get('class_1', 0.0)
                    adv_forgery_acc = acc_details['adversarial'].get('class_1', 0.0)
                    
                    # 計算變化率
                    if orig_forgery_acc > 0:
                        change_rate = (adv_forgery_acc - orig_forgery_acc) / orig_forgery_acc * 100
                    else:
                        change_rate = 0.0
                        
                    print(f"  偽造區域準確率: {orig_forgery_acc:.4f} → {adv_forgery_acc:.4f} (變化率: {change_rate:.2f}%)")
                
            # 如果有類別混淆信息，顯示主要的混淆類別
            if 'Class_Confusion' in seg_metrics:
                confusion = seg_metrics['Class_Confusion']
                
                if isinstance(confusion, dict) and confusion:
                    print("  類別混淆分析:")
                    
                    # 僅顯示前3個主要混淆
                    if 'prediction_change' in confusion:
                        # 找出非對角線元素中最大的3個值
                        matrix = np.array(confusion['prediction_change'])
                        off_diag = []
                        for i in range(matrix.shape[0]):
                            for j in range(matrix.shape[1]):
                                if i != j:  # 非對角線元素
                                    off_diag.append((i, j, matrix[i, j]))
                        
                        # 按值排序，取前3個
                        sorted_confusions = sorted(off_diag, key=lambda x: x[2], reverse=True)[:3]
                        
                        for from_cls, to_cls, ratio in sorted_confusions:
                            print(f"    類別 {from_cls} → 類別 {to_cls}: {ratio*100:.2f}%")
    
    # 保存原始和對抗樣本
    if args.save_images:
        logging.info("保存圖像樣本...")
        
        # 選擇一部分樣本進行保存
        num_samples = min(5, clean_images.size(0))
        for i in range(num_samples):
            clean_img = clean_images[i].cpu().numpy()
            adv_img = adv_images[i].cpu().numpy()
            pert_img = perturbations[i].cpu().numpy()
            
            # 文件路徑
            sample_dir = os.path.join(images_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存原始和對抗樣本數據
            clean_npy_path = os.path.join(sample_dir, "clean.npy")
            adv_npy_path = os.path.join(sample_dir, "adversarial.npy")
            pert_npy_path = os.path.join(sample_dir, "perturbation.npy")
            
            np.save(clean_npy_path, clean_img)
            np.save(adv_npy_path, adv_img)
            np.save(pert_npy_path, pert_img)
            
            # 保存視覺化結果
            vis_path = os.path.join(sample_dir, "visualization.png")
            visualize_adversarial(
                clean_img, 
                adv_img, 
                label=labels_list[i].cpu().numpy(),  # 傳遞標籤作為Ground Truth
                pred_orig=clean_preds[i].cpu().numpy() if clean_preds.size(0) > i else None, 
                pred_adv=adv_preds[i].cpu().numpy() if adv_preds.size(0) > i else None,
                output_path=vis_path
            )
            
            # 保存簡化的樣本元數據
            metadata = {
                "樣本索引": i,
                "L2範數": float(np.sqrt(np.sum((clean_img - adv_img) ** 2)))
            }
            
            metadata_path = os.path.join(sample_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        logging.info(f"已保存 {num_samples} 個樣本圖像到: {images_dir}")
    
    # 保存擾動
    if args.save_perturbations:
        logging.info("保存擾動...")
        pert_file = os.path.join(perturbations_dir, "perturbations.pt")
        torch.save(perturbations, pert_file)
        
        # 保存簡化的擾動統計信息
        pert_stats = {
            "樣本數量": perturbations.size(0),
            "平均L2範數": float((perturbations ** 2).sum(dim=(1, 2, 3)).sqrt().mean()),
            "平均L∞範數": float(perturbations.abs().amax(dim=(1, 2, 3)).mean())
        }
        
        pert_stats_path = os.path.join(perturbations_dir, "perturbation_stats.json")
        with open(pert_stats_path, 'w', encoding='utf-8') as f:
            json.dump(pert_stats, f, ensure_ascii=False, indent=4)
            
        logging.info(f"擾動已保存到: {pert_file}")
    
    # 保存成功攻擊的樣本索引
    success_indices = []
    for i in range(len(clean_preds)):
        # 檢查是否所有像素的預測都相同，若有任何不同，則攻擊成功
        if torch.any(clean_preds[i] != adv_preds[i]).item():
            success_indices.append(i)
    
    success_file = os.path.join(result_dir, "successful_attacks.json")
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump({
            "成功樣本數": len(success_indices),
            "總樣本數": len(clean_preds),
            "成功率": len(success_indices) / len(clean_preds) if len(clean_preds) > 0 else 0
        }, f, ensure_ascii=False, indent=4)
    
    logging.info(f"攻擊完成，結果已保存到: {result_dir}")


# 添加一個 JSON 編碼器來處理 NumPy 數據類型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    main() 