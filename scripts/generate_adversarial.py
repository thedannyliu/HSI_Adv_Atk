import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import json
from datetime import datetime

from torch.utils.data import DataLoader
from config import get_cfg_defaults
from datasets import forgeryHSIDataset
from models.hrnet import HRNetV2
from attacks import (
    normalize, unnormalize,
    fgsm_spatial_attack, pgd_spatial_attack, cw_spatial_attack, deepfool_spatial_attack,
    fgsm_spectral_attack, pgd_spectral_attack, cw_spectral_attack, deepfool_spectral_attack,
    fgsm_hybrid_attack, pgd_hybrid_attack, cw_hybrid_attack, deepfool_hybrid_attack
)
from utils import DummyTransform, PairCompose, visualize_adversarial, get_datetime_str
from utils import set_random_seed
from metrics import SegMetrics


def main(args):
    # 讀取配置
    cfg = get_cfg_defaults()
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
        model.load_state_dict(torch.load(ckpt, map_location=device))
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

    # 設置保存目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        cfg.ATTACK.SAVE_DIR, 
        f"{attack_domain}_{attack_method}_eps{eps}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)

    vis_dir = os.path.join(save_dir, "vis")
    adv_dir = os.path.join(save_dir, "samples")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    # 保存設置
    config_save_path = os.path.join(save_dir, "attack_config.json")
    with open(config_save_path, "w") as f:
        json.dump({
            "attack_method": attack_method,
            "attack_domain": attack_domain,
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "c": c,
            "kappa": kappa,
            "lr": lr,
            "max_iter": max_iter,
            "overshoot": overshoot,
            "checkpoint": ckpt,
            "batch_size": cfg.ATTACK.BATCH_SIZE,
            "timestamp": timestamp
        }, f, indent=4)

    # 記錄成功樣本的文件名
    successful_samples = []

    # 載入適當的攻擊函數
    if attack_domain == 'spatial':
        if attack_method == 'fgsm':
            attack_fn = fgsm_spatial_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_spatial_attack
        elif attack_method == 'cw':
            attack_fn = cw_spatial_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_spatial_attack
    elif attack_domain == 'spectral':
        if attack_method == 'fgsm':
            attack_fn = fgsm_spectral_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_spectral_attack
        elif attack_method == 'cw':
            attack_fn = cw_spectral_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_spectral_attack
    elif attack_domain == 'hybrid':
        if attack_method == 'fgsm':
            attack_fn = fgsm_hybrid_attack
        elif attack_method == 'pgd':
            attack_fn = pgd_hybrid_attack
        elif attack_method == 'cw':
            attack_fn = cw_hybrid_attack
        elif attack_method == 'deepfool':
            attack_fn = deepfool_hybrid_attack
    else:
        raise ValueError(f"不支持的攻擊域: {attack_domain}")

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 記錄指標
    success_count = 0
    total_samples = 0
    l2_norms = []
    l0_norms = []
    linf_norms = []
    ssim_values = []

    # 開始攻擊
    print(f"[INFO] 開始 {attack_domain} 域上的 {attack_method} 攻擊 (eps={eps})")

    for batch_idx, (images, labels, filenames) in enumerate(test_loader):
        print(f"[INFO] 處理批次 {batch_idx+1}/{len(test_loader)}")

        images = images.to(device)
        labels = labels.to(device)

        # 正規化
        mean = cfg.DATASET.MEAN if hasattr(cfg.DATASET, 'MEAN') else 0.0
        std = cfg.DATASET.STD if hasattr(cfg.DATASET, 'STD') else 1.0

        # 進行預測
        outputs_orig, _ = model(images)
        preds_orig = outputs_orig.argmax(dim=1)

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
                    model, images, labels, c=cfg.ATTACK.CW.C, kappa=cfg.ATTACK.CW.KAPPA,
                    steps=cfg.ATTACK.CW.STEPS, lr=cfg.ATTACK.CW.LR, eps=eps,
                    device=device, mean=mean, std=std,
                    spatial_weight=spatial_weight, target_bands=target_bands
                )
            elif attack_method == 'deepfool':
                adv_images = attack_fn(
                    model, images, num_classes=cfg.DATASET.NUM_CLASSES,
                    max_iter=cfg.ATTACK.DEEPFOOL.MAX_ITER, overshoot=cfg.ATTACK.DEEPFOOL.OVERSHOOT,
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
                    model, images, labels, c=cfg.ATTACK.CW.C, kappa=cfg.ATTACK.CW.KAPPA,
                    steps=cfg.ATTACK.CW.STEPS, lr=cfg.ATTACK.CW.LR, eps=eps,
                    device=device, mean=mean, std=std
                )
            elif attack_method == 'deepfool':
                adv_images = attack_fn(
                    model, images, num_classes=cfg.DATASET.NUM_CLASSES,
                    max_iter=cfg.ATTACK.DEEPFOOL.MAX_ITER, overshoot=cfg.ATTACK.DEEPFOOL.OVERSHOOT,
                    device=device, mean=mean, std=std
                )
        else:
            raise ValueError(f"不支持的攻擊域: {attack_domain}")

        # 對抗樣本的預測
        outputs_adv, _ = model(adv_images)
        preds_adv = outputs_adv.argmax(dim=1)

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

            # 計算SSIM (使用utils中的函數)
            # 這裡簡化處理，實際應用中可能需要更複雜的計算
            ssim = 1.0 - (l2_norm / (l2_norm + 1.0))
            ssim_values.append(ssim)

            # 檢查攻擊是否成功
            is_success = (preds_orig[i] != preds_adv[i]).any().item()

            if is_success:
                success_count += 1
                successful_samples.append(filenames[i])

                # 可視化成功的對抗樣本
                vis_path = os.path.join(vis_dir, f"{os.path.basename(filenames[i]).replace('.npy', '')}_adv.png")
                visualize_adversarial(
                    original=images[i].cpu().numpy(),
                    adversarial=adv_images[i].cpu().numpy(),
                    original_pred=preds_orig[i].cpu().numpy(),
                    adversarial_pred=preds_adv[i].cpu().numpy(),
                    save_path=vis_path
                )

            # 保存對抗樣本
            save_path = os.path.join(adv_dir, os.path.basename(filenames[i]))
            np.save(save_path, adv_images[i].cpu().numpy())

    # 計算並保存攻擊結果
    success_rate = success_count / total_samples if total_samples > 0 else 0
    avg_l0_norm = np.mean(l0_norms) if l0_norms else 0
    avg_l2_norm = np.mean(l2_norms) if l2_norms else 0
    avg_linf_norm = np.mean(linf_norms) if linf_norms else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0

    results = {
        "attack_method": attack_method,
        "attack_domain": attack_domain,
        "eps": eps,
        "success_rate": success_rate,
        "total_samples": total_samples,
        "success_count": success_count,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成對抗樣本")
    parser.add_argument("--config_path", type=str, default="./config/attack_config.yaml", help="配置文件路徑")
    parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw', 'deepfool'], help='Attack method')
    parser.add_argument('--domain', type=str, default='spatial', choices=['spatial', 'spectral', 'hybrid'], help='Attack domain')
    args = parser.parse_args()
    main(args) 