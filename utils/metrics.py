import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips  # 新增: LPIPS計算
from torch_fidelity import calculate_metrics  # 新增: 更全面的圖像質量評估

# 全局LPIPS模型
lpips_model = None

def get_lpips_model(device='cpu'):
    """獲取LPIPS模型單例"""
    global lpips_model
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net='alex').to(device)
    return lpips_model

class SegMetrics:
    """
    Computes segmentation metrics based on the confusion matrix.
    Matches the implementation in DFCN_Py_all/DFCN_Py/HRNetV2/metrics/metrics.py
    """
    
    def __init__(self, n_class, device):
        self.n_class = n_class
        self.device = device # Store device for resetting
        self.confusion_matrix = torch.zeros((n_class, n_class), dtype=torch.int64).to(device)
        
    def update(self, true, pred):
        """
        Update the confusion matrix.
        Args:
            true: Ground truth labels (B, H, W), LongTensor.
            pred: Predicted labels (B, H, W), LongTensor.
        """
        for t, p in zip(true, pred):
            self.confusion_matrix += self._fast_hist(t.flatten(), p.flatten())
            
    def _fast_hist(self, true, pred):
        """
        Compute the confusion matrix for a flattened label pair.
        Args:
            true: Flattened ground truth labels.
            pred: Flattened predicted labels.
        Returns:
            Confusion matrix update (n_class, n_class).
        """
        mask = (true >= 0) & (true < self.n_class)
        # Ensure indices are within bounds before calculating bin counts
        true_masked = true[mask].to(dtype=torch.int64)
        pred_masked = pred[mask].to(dtype=torch.int64)
        # Calculate index for bincount: label_true * n_class + label_pred
        hist = torch.bincount(
            self.n_class * true_masked + pred_masked, 
            minlength=self.n_class**2
        ).reshape(self.n_class, self.n_class)
        return hist
    
    def get_results(self):
        """
        Compute metrics from the accumulated confusion matrix.
        Returns:
            Dictionary containing overall accuracy, mean accuracy, mean IoU, and class IoU.
        """
        hist = self.confusion_matrix.float() # Use float for calculations
        
        # Overall Accuracy
        acc = torch.diag(hist).sum() / hist.sum()
        
        # Per-Class Accuracy and Mean Accuracy
        acc_cls = torch.diag(hist) / hist.sum(dim=1) # Accuracy for each class
        # Handle cases where a class has no samples (division by zero -> NaN)
        acc_cls_nanmean = torch.nanmean(acc_cls) # Calculate mean ignoring NaNs
        if torch.isnan(acc_cls_nanmean): # If all are NaN (no samples in any class?)
             acc_cls_nanmean = torch.tensor(0.0, device=self.device)
             
        # Per-Class IoU and Mean IoU
        iou = torch.diag(hist) / (hist.sum(dim=0) + hist.sum(dim=1) - torch.diag(hist))
        mean_iou_nanmean = torch.nanmean(iou) # Calculate mean ignoring NaNs
        if torch.isnan(mean_iou_nanmean): # If all are NaN
             mean_iou_nanmean = torch.tensor(0.0, device=self.device)
             
        # Class IoU dictionary
        cls_iou = {i: iou[i].item() if not torch.isnan(iou[i]) else 0.0 for i in range(self.n_class)}
        
        # 計算每個類別的F1分數
        f1_scores = {}
        for i in range(self.n_class):
            if not torch.isnan(iou[i]) and iou[i] > 0:
                # F1 = 2 * IoU / (1 + IoU)
                f1_scores[i] = (2 * iou[i] / (1 + iou[i])).item()
            else:
                f1_scores[i] = 0.0
                
        # 計算平均F1分數
        mean_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
        
        return {
            "Overall acc": acc.item() if not torch.isnan(acc) else 0.0,
            "Mean acc": acc_cls_nanmean.item(),
            "Mean IoU": mean_iou_nanmean.item(),
            "Class IoU": cls_iou,
            "Class F1": f1_scores,
            "Mean F1": mean_f1
        }
        
    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = torch.zeros((self.n_class, self.n_class), dtype=torch.int64).to(self.device)

# --- Keep existing functions if they are still needed, otherwise they can be removed --- 
# (calculate_miou and calculate_accuracy are effectively replaced by SegMetrics.get_results()['Mean IoU'] and ['Overall acc'])

def calculate_miou(pred, target, debug=False):
    """
    Calculates Mean Intersection over Union (mIoU) for binary case.
    Kept for potential compatibility, but SegMetrics is preferred.
    Args:
        pred (torch.Tensor): Predictions (B, H, W), Long or Float.
        target (torch.Tensor): Ground truth (B, H, W), Long.
    Returns:
        float: Mean IoU.
    """
    # Use SegMetrics for consistent calculation
    metric = SegMetrics(n_class=2, device=pred.device) 
    pred_long = (pred > 0.5).long() if pred.is_floating_point() else pred.long()
    metric.update(target.long(), pred_long)
    results = metric.get_results()
    return results["Mean IoU"] 

def calculate_accuracy(pred, target, debug=False):
    """
    Calculates pixel accuracy for binary case.
    Kept for potential compatibility, but SegMetrics is preferred.
     Args:
        pred (torch.Tensor): Predictions (B, H, W), Long or Float.
        target (torch.Tensor): Ground truth (B, H, W), Long.
    Returns:
        float: Accuracy.
    """
    # Use SegMetrics for consistent calculation
    metric = SegMetrics(n_class=2, device=pred.device) 
    pred_long = (pred > 0.5).long() if pred.is_floating_point() else pred.long()
    metric.update(target.long(), pred_long)
    results = metric.get_results()
    return results["Overall acc"] 

def calculate_spectral_perturbation_impact(original, perturbed, n_bands=None):
    """
    計算光譜域上的擾動影響
    
    Args:
        original (numpy.ndarray): 原始高光譜圖像 [C, H, W]
        perturbed (numpy.ndarray): 對抗高光譜圖像 [C, H, W]
        n_bands (int, optional): 要分析的波段數，若為None則使用所有波段
        
    Returns:
        dict: 包含光譜擾動影響分析結果
    """
    if original.shape != perturbed.shape:
        raise ValueError("原始圖像和對抗圖像的形狀必須相同")
    
    C, H, W = original.shape
    
    # 計算每個波段的平均擾動
    band_perturbations = []
    for i in range(C):
        band_orig = original[i]
        band_pert = perturbed[i]
        
        # 計算波段擾動
        abs_diff = np.abs(band_pert - band_orig)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        
        band_perturbations.append({
            "band_idx": i,
            "mean_diff": float(mean_diff),
            "max_diff": float(max_diff),
            "relative_diff": float(mean_diff / (np.mean(np.abs(band_orig)) + 1e-8))
        })
    
    # 按平均擾動大小排序
    sorted_bands = sorted(band_perturbations, key=lambda x: x["mean_diff"], reverse=True)
    
    # 若指定了n_bands，則取擾動最大的n_bands個波段
    if n_bands is not None and n_bands < C:
        top_bands = sorted_bands[:n_bands]
    else:
        top_bands = sorted_bands
    
    # 計算整體光譜擾動的統計量
    total_mean_diff = np.mean([b["mean_diff"] for b in band_perturbations])
    total_max_diff = np.max([b["max_diff"] for b in band_perturbations])
    
    return {
        "spectral_impact": {
            "mean_perturbation": float(total_mean_diff),
            "max_perturbation": float(total_max_diff),
            "most_affected_bands": [b["band_idx"] for b in top_bands[:10]]  # 列出受影響最大的10個波段
        },
        "band_perturbations": sorted_bands
    }

class AdvMetrics:
    """
    計算對抗攻擊的評估指標
    """
    def __init__(self, device, significant_pixel_ratio=0.05):
        self.device = device
        self.total_samples = 0
        self.success_count = 0  # 一般成功定義：任何像素變化
        self.significant_success_count = 0  # 顯著成功：超過閾值比例的像素變化
        self.significant_pixel_ratio = significant_pixel_ratio  # 判定顯著變化的閾值比例
        
        # 對每個類別的攻擊成功次數
        self.class_success_counts = {}
        
        # 擾動範數統計
        self.l0_norms = []
        self.l1_norms = []
        self.l2_norms = []
        self.linf_norms = []
        
        # 圖像質量指標
        self.ssim_values = []
        self.psnr_values = []
        self.lpips_values = []  # 新增: LPIPS值
        
        # 預測置信度變化
        self.confidence_changes = []
        
        # 混淆矩陣狀態追蹤
        self.pred_changes = {}  # 從類別A變為類別B的像素數量
        
        # 光譜擾動分析
        self.spectral_impacts = []
        
        # 初始化LPIPS模型
        self.lpips_model = get_lpips_model(device)
        
    def update(self, pred_orig, pred_adv, images_orig, images_adv, labels):
        """
        更新對抗攻擊指標
        Args:
            pred_orig: 原始圖像的預測結果 (B, H, W)
            pred_adv: 對抗樣本的預測結果 (B, H, W)
            images_orig: 原始圖像 (B, C, H, W)
            images_adv: 對抗樣本 (B, C, H, W)
            labels: 真實標籤 (B, H, W)
        """
        batch_size = images_orig.size(0)
        self.total_samples += batch_size
        
        # 計算擾動範數
        perturbation = images_adv - images_orig
        perturbation_np = perturbation.detach().cpu().numpy()
        
        for i in range(batch_size):
            # 計算像素級變化
            mask = (pred_orig[i] != pred_adv[i])
            changed_pixels = mask.sum().item()
            total_pixels = mask.numel()
            
            # 1. 標準攻擊成功定義：任何像素發生變化
            if changed_pixels > 0:
                self.success_count += 1
                
            # 2. 顯著攻擊成功定義：超過閾值比例的像素發生變化
            changed_ratio = changed_pixels / total_pixels
            if changed_ratio >= self.significant_pixel_ratio:
                self.significant_success_count += 1
                
            # 3. 對重要區域（偽造區域）的攻擊效果
            # 計算僅在有標籤的區域（偽造區域）發生的變化
            for cls in range(labels[i].max().item() + 1):
                class_mask = (labels[i] == cls)
                class_changes = (pred_orig[i] != pred_adv[i]) & class_mask
                class_changed_pixels = class_changes.sum().item()
                class_total_pixels = class_mask.sum().item()
                
                if cls not in self.class_success_counts:
                    self.class_success_counts[cls] = {"count": 0, "total": 0}
                
                if class_total_pixels > 0 and class_changed_pixels > 0:
                    self.class_success_counts[cls]["count"] += 1
                if class_total_pixels > 0:
                    self.class_success_counts[cls]["total"] += 1
            
            # 4. 混淆矩陣變化統計
            for from_class in range(labels[i].max().item() + 1):
                for to_class in range(labels[i].max().item() + 1):
                    if from_class == to_class:
                        continue
                        
                    # 計算從A類預測變為B類的像素數量
                    from_to_key = f"{from_class}_to_{to_class}"
                    change_mask = (pred_orig[i] == from_class) & (pred_adv[i] == to_class)
                    change_count = change_mask.sum().item()
                    
                    if from_to_key not in self.pred_changes:
                        self.pred_changes[from_to_key] = 0
                    self.pred_changes[from_to_key] += change_count
            
            # 5. 擾動範數計算
            # L0範數（非零元素數量）
            l0_norm = np.count_nonzero(perturbation_np[i]) / perturbation_np[i].size
            self.l0_norms.append(l0_norm)
            
            # L1範數
            l1_norm = np.sum(np.abs(perturbation_np[i]))
            self.l1_norms.append(l1_norm)
            
            # L2範數
            l2_norm = np.sqrt(np.sum(perturbation_np[i]**2))
            self.l2_norms.append(l2_norm)
            
            # L∞範數
            linf_norm = np.max(np.abs(perturbation_np[i]))
            self.linf_norms.append(linf_norm)
            
            # 6. 圖像質量評估
            # 確保形狀正確
            orig_img = images_orig[i].cpu().numpy()
            adv_img = images_adv[i].cpu().numpy()
            
            # SSIM計算
            try:
                ssim_value = ssim(orig_img, adv_img, channel_axis=0, data_range=1.0)
                self.ssim_values.append(ssim_value)
            except Exception as e:
                print(f"SSIM計算錯誤：{str(e)}")
                
            # PSNR計算
            try:
                psnr_value = psnr(orig_img, adv_img, data_range=1.0)
                self.psnr_values.append(psnr_value)
            except Exception as e:
                print(f"PSNR計算錯誤：{str(e)}")
            
            # 7. LPIPS計算
            try:
                # 準備LPIPS輸入：需調整到[0,1]範圍的RGB張量，形狀為1x3xHxW
                if images_orig[i].size(0) >= 3:  # 確保有3個或更多通道
                    # 使用最大方差的3個通道作為RGB
                    variances = torch.var(images_orig[i], dim=(1, 2))
                    _, indices = torch.topk(variances, 3)
                    
                    # 從高光譜圖像選擇三個通道作為RGB
                    orig_rgb = torch.zeros(3, images_orig[i].size(1), images_orig[i].size(2), device=self.device)
                    adv_rgb = torch.zeros(3, images_adv[i].size(1), images_adv[i].size(2), device=self.device)
                    
                    for j, idx in enumerate(indices):
                        orig_rgb[j] = images_orig[i][idx]
                        adv_rgb[j] = images_adv[i][idx]
                    
                    # 確保數據範圍在[0,1]
                    orig_rgb = torch.clamp(orig_rgb, 0, 1)
                    adv_rgb = torch.clamp(adv_rgb, 0, 1)
                    
                    # 轉為LPIPS期望的格式：1x3xHxW
                    orig_rgb = orig_rgb.unsqueeze(0)
                    adv_rgb = adv_rgb.unsqueeze(0)
                    
                    # 計算LPIPS值
                    with torch.no_grad():
                        lpips_value = self.lpips_model(orig_rgb, adv_rgb).item()
                    self.lpips_values.append(lpips_value)
            except Exception as e:
                print(f"LPIPS計算錯誤：{str(e)}")
                
            # 8. 光譜擾動分析
            try:
                # 進行光譜擾動分析 
                spectral_impact = calculate_spectral_perturbation_impact(
                    original=orig_img, 
                    perturbed=adv_img,
                    n_bands=10  # 分析影響最大的10個波段
                )
                self.spectral_impacts.append(spectral_impact)
            except Exception as e:
                print(f"光譜擾動分析錯誤：{str(e)}")
    
    def get_results(self):
        """
        獲取對抗攻擊的評估結果
        Returns:
            Dictionary containing enhanced attack metrics
        """
        if self.total_samples == 0:
            return {
                "Attack_Success_Rate": 0.0,
                "Significant_Attack_Success_Rate": 0.0,
                "Class_Attack_Success_Rate": {},
                "Average_L0_Norm": 0.0,
                "Average_L1_Norm": 0.0,
                "Average_L2_Norm": 0.0,
                "Average_Linf_Norm": 0.0,
                "Average_SSIM": 0.0,
                "Average_PSNR": 0.0,
                "Average_LPIPS": 0.0,
                "Prediction_Changes": {},
                "Spectral_Impact": {}
            }
        
        # 計算每個類別的攻擊成功率
        class_success_rates = {}
        for cls, data in self.class_success_counts.items():
            if data["total"] > 0:
                class_success_rates[str(cls)] = data["count"] / data["total"]
            else:
                class_success_rates[str(cls)] = 0.0
        
        # 彙總光譜擾動分析
        spectral_impact_summary = {}
        if self.spectral_impacts:
            # 計算各波段平均擾動
            all_band_perturbations = {}
            
            for impact in self.spectral_impacts:
                for band_info in impact.get("band_perturbations", []):
                    band_idx = band_info["band_idx"]
                    if band_idx not in all_band_perturbations:
                        all_band_perturbations[band_idx] = []
                    all_band_perturbations[band_idx].append(band_info["mean_diff"])
            
            # 計算平均值
            avg_band_perturbations = {}
            for band_idx, values in all_band_perturbations.items():
                avg_band_perturbations[band_idx] = sum(values) / len(values)
            
            # 找出平均擾動最大的波段
            sorted_bands = sorted(avg_band_perturbations.items(), key=lambda x: x[1], reverse=True)
            most_affected_bands = [int(band) for band, _ in sorted_bands[:10]]
            
            spectral_impact_summary = {
                "most_affected_bands": most_affected_bands,
                "band_perturbation_summary": dict(sorted_bands[:20])  # 列出前20個波段的擾動
            }
        
        return {
            # 1. 攻擊成功率指標
            "Attack_Success_Rate": self.success_count / self.total_samples,
            "Significant_Attack_Success_Rate": self.significant_success_count / self.total_samples,
            "Class_Attack_Success_Rate": class_success_rates,
            
            # 2. 擾動範數指標
            "Average_L0_Norm": np.mean(self.l0_norms) if self.l0_norms else 0.0,
            "Average_L1_Norm": np.mean(self.l1_norms) if self.l1_norms else 0.0,
            "Average_L2_Norm": np.mean(self.l2_norms) if self.l2_norms else 0.0,
            "Average_Linf_Norm": np.mean(self.linf_norms) if self.linf_norms else 0.0,
            
            # 3. 圖像質量指標
            "Average_SSIM": np.mean(self.ssim_values) if self.ssim_values else 0.0,
            "Average_PSNR": np.mean(self.psnr_values) if self.psnr_values else 0.0,
            "Average_LPIPS": np.mean(self.lpips_values) if self.lpips_values else 0.0,
            
            # 4. 混淆矩陣變化指標
            "Prediction_Changes": self.pred_changes,
            
            # 5. 光譜擾動分析
            "Spectral_Impact": spectral_impact_summary
        }
    
    def reset(self):
        """重置所有指標"""
        self.total_samples = 0
        self.success_count = 0
        self.significant_success_count = 0
        self.class_success_counts = {}
        self.l0_norms = []
        self.l1_norms = []
        self.l2_norms = []
        self.linf_norms = []
        self.ssim_values = []
        self.psnr_values = []
        self.lpips_values = []
        self.confidence_changes = []
        self.pred_changes = {}
        self.spectral_impacts = []