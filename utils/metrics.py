import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips  # 新增: LPIPS計算
from torch_fidelity import calculate_metrics  # 新增: 更全面的圖像質量評估
import warnings
import torch.nn.functional as F

# 全局LPIPS模型
lpips_model = None

def get_lpips_model(device='cpu'):
    """獲取LPIPS模型單例"""
    global lpips_model
    if lpips_model is None:
        # 使用警告抑制來避免顯示棄用警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

def calculate_spectral_perturbation_impact(original, perturbed):
    """
    計算光譜擾動對原始樣本的影響
    
    Args:
        original (numpy.ndarray or torch.Tensor): 原始樣本，形狀為 [C, H, W] 或 [B, C, H, W]
        perturbed (numpy.ndarray or torch.Tensor): 對抗樣本，形狀為 [C, H, W] 或 [B, C, H, W]
        
    Returns:
        dict: 包含不同指標的字典
    """
    try:
        # 將Tensor轉為NumPy數組
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.detach().cpu().numpy()
        
        # 處理批次情況
        if original.ndim == 4:  # [B, C, H, W]
            results = []
            for i in range(original.shape[0]):
                results.append(calculate_spectral_perturbation_impact(original[i], perturbed[i]))
            
            # 合併結果
            merged_results = {}
            for key in results[0].keys():
                if isinstance(results[0][key], (int, float)):
                    merged_results[key] = np.mean([r[key] for r in results])
                else:
                    # 對於數組類型，逐元素平均
                    merged_results[key] = np.mean([r[key] for r in results], axis=0).tolist()
            
            return merged_results
        
        # 確保是三維數組：[C, H, W]
        if original.ndim != 3 or perturbed.ndim != 3:
            raise ValueError(f"輸入維度不正確: original {original.shape}, perturbed {perturbed.shape}")
        
        # 獲取形狀
        C, H, W = original.shape
        
        # 確保輸入範圍在 [0, 1]
        original = np.clip(original, 0, 1)
        perturbed = np.clip(perturbed, 0, 1)
        
        # 計算每個波段的平均變化
        bands_differences = np.mean(np.abs(original - perturbed), axis=(1, 2))
        
        # 計算各種指標
        max_band_diff = np.max(bands_differences)
        avg_band_diff = np.mean(bands_differences)
        std_band_diff = np.std(bands_differences)
        
        # 波段敏感性：哪些波段變化最大
        top_affected_bands = np.argsort(bands_differences)[::-1][:10].tolist()  # 受影響最大的10個波段
        
        # 相對波段變化
        relative_band_impact = bands_differences / (np.mean(original, axis=(1, 2)) + 1e-8)
        
        # 計算影響分布
        # 波長分佈：假設波段是均勻分布的
        wavelength_distribution = np.arange(C)
        # 波長敏感性指標 - 修复除以零的问题
        sum_diff = np.sum(bands_differences)
        if sum_diff > 1e-8:  # 確保不會除以零
            wavelength_sensitivity = np.sum(bands_differences * wavelength_distribution) / sum_diff
        else:
            wavelength_sensitivity = 0.0
        
        # 結果字典
        result = {
            'max_band_difference': float(max_band_diff),
            'avg_band_difference': float(avg_band_diff),
            'std_band_difference': float(std_band_diff),
            'top_affected_bands': top_affected_bands,
            'relative_band_impact': relative_band_impact.tolist(),
            'wavelength_sensitivity': float(wavelength_sensitivity),
            'band_differences': bands_differences.tolist()
        }
        
        return result
    except Exception as e:
        print(f"光譜擾動分析錯誤：{str(e)}")
        # 返回空結果
        return {
            'max_band_difference': 0.0,
            'avg_band_difference': 0.0,
            'std_band_difference': 0.0,
            'top_affected_bands': [],
            'relative_band_impact': [],
            'wavelength_sensitivity': 0.0,
            'band_differences': []
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
        
        # 保存每個樣本的詳細指標，用於對比不同擾動配置
        self.sample_metrics = []
        
    def update(self, images_orig, images_adv, labels, pred_orig, pred_adv, attack_config=None):
        """
        更新對抗攻擊指標
        Args:
            images_orig: 原始圖像 (B, C, H, W)
            images_adv: 對抗樣本 (B, C, H, W)
            labels: 真實標籤 (B, H, W)
            pred_orig: 原始圖像的預測結果 (B, H, W)
            pred_adv: 對抗樣本的預測結果 (B, H, W)
            attack_config: 當前攻擊配置，可選，用於保存樣本級別的評估結果
        """
        batch_size = images_orig.size(0)
        self.total_samples += batch_size
        
        # 計算擾動範數
        perturbation = images_adv - images_orig
        perturbation_np = perturbation.detach().cpu().numpy()
        
        for i in range(batch_size):
            # 初始化當前樣本的評估結果
            sample_result = {}
            if attack_config:
                sample_result["attack_config"] = attack_config
            
            # 計算像素級變化
            mask = (pred_orig[i] != pred_adv[i])
            changed_pixels = mask.sum().item()
            total_pixels = mask.numel()
            
            # 1. 標準攻擊成功定義：任何像素發生變化
            is_success = changed_pixels > 0
            if is_success:
                self.success_count += 1
                sample_result["success"] = True
            else:
                sample_result["success"] = False
                
            # 2. 顯著攻擊成功定義：超過閾值比例的像素發生變化
            changed_ratio = changed_pixels / total_pixels
            sample_result["changed_ratio"] = changed_ratio
            
            if changed_ratio >= self.significant_pixel_ratio:
                self.significant_success_count += 1
                sample_result["significant_success"] = True
            else:
                sample_result["significant_success"] = False
                
            # 3. 對重要區域（偽造區域）的攻擊效果
            class_changes = {}
            # 計算僅在有標籤的區域（偽造區域）發生的變化
            for cls in range(labels[i].max().item() + 1):
                class_mask = (labels[i] == cls)
                class_changes_mask = (pred_orig[i] != pred_adv[i]) & class_mask
                class_changed_pixels = class_changes_mask.sum().item()
                class_total_pixels = class_mask.sum().item()
                
                class_changes[int(cls)] = {
                    "changed_pixels": int(class_changed_pixels),
                    "total_pixels": int(class_total_pixels),
                    "changed_ratio": float(class_changed_pixels / class_total_pixels if class_total_pixels > 0 else 0)
                }
                
                if cls not in self.class_success_counts:
                    self.class_success_counts[cls] = {"count": 0, "total": 0}
                
                if class_total_pixels > 0 and class_changed_pixels > 0:
                    self.class_success_counts[cls]["count"] += 1
                if class_total_pixels > 0:
                    self.class_success_counts[cls]["total"] += 1
            
            sample_result["class_changes"] = class_changes
            
            # 4. 混淆矩陣變化統計
            pred_changes = {}
            for from_class in range(labels[i].max().item() + 1):
                for to_class in range(labels[i].max().item() + 1):
                    if from_class == to_class:
                        continue
                        
                    # 計算從A類預測變為B類的像素數量
                    from_to_key = f"{from_class}_to_{to_class}"
                    change_mask = (pred_orig[i] == from_class) & (pred_adv[i] == to_class)
                    change_count = change_mask.sum().item()
                    
                    pred_changes[from_to_key] = int(change_count)
                    
                    if from_to_key not in self.pred_changes:
                        self.pred_changes[from_to_key] = 0
                    self.pred_changes[from_to_key] += change_count
            
            sample_result["prediction_changes"] = pred_changes
            
            # 5. 擾動範數計算
            # L0範數（非零元素數量）
            l0_norm = np.count_nonzero(perturbation_np[i]) / perturbation_np[i].size
            self.l0_norms.append(l0_norm)
            sample_result["l0_norm"] = float(l0_norm)
            
            # L1範數
            l1_norm = np.sum(np.abs(perturbation_np[i]))
            self.l1_norms.append(l1_norm)
            sample_result["l1_norm"] = float(l1_norm)
            
            # L2範數
            l2_norm = np.sqrt(np.sum(perturbation_np[i]**2))
            self.l2_norms.append(l2_norm)
            sample_result["l2_norm"] = float(l2_norm)
            
            # L∞範數
            linf_norm = np.max(np.abs(perturbation_np[i]))
            self.linf_norms.append(linf_norm)
            sample_result["linf_norm"] = float(linf_norm)
            
            # 6. 圖像質量評估
            # 確保形狀正確
            orig_img = images_orig[i].detach().cpu().numpy()
            adv_img = images_adv[i].detach().cpu().numpy()
            
            # SSIM計算
            try:
                # 選擇方差最大的三個通道進行 SSIM 計算
                if orig_img.shape[0] > 3:
                    variances = np.var(orig_img, axis=(1, 2))
                    top_channels = np.argsort(variances)[-3:]
                    
                    # 創建RGB顯示圖像 (3, H, W)
                    orig_rgb = np.zeros((3, orig_img.shape[1], orig_img.shape[2]))
                    adv_rgb = np.zeros((3, adv_img.shape[1], adv_img.shape[2]))
                    
                    for j, channel in enumerate(top_channels):
                        orig_rgb[j] = orig_img[channel]
                        adv_rgb[j] = adv_img[channel]
                    
                    # 規範化到0-1範圍
                    orig_rgb = (orig_rgb - orig_rgb.min()) / (orig_rgb.max() - orig_rgb.min() + 1e-8)
                    adv_rgb = (adv_rgb - adv_rgb.min()) / (adv_rgb.max() - adv_rgb.min() + 1e-8)
                    
                    # 使用 RGB 版本計算 SSIM
                    ssim_value = ssim(orig_rgb, adv_rgb, channel_axis=0, data_range=1.0)
                else:
                    ssim_value = ssim(orig_img, adv_img, channel_axis=0, data_range=1.0)
                
                self.ssim_values.append(ssim_value)
                sample_result["ssim"] = float(ssim_value)
            except Exception as e:
                print(f"SSIM計算錯誤：{str(e)}")
                sample_result["ssim"] = 0.0
                
            # PSNR計算
            try:
                # 同樣選擇方差最大的三個通道進行 PSNR 計算
                if orig_img.shape[0] > 3 and 'orig_rgb' in locals():
                    psnr_value = psnr(orig_rgb, adv_rgb, data_range=1.0)
                else:
                    psnr_value = psnr(orig_img, adv_img, data_range=1.0)
                
                self.psnr_values.append(psnr_value)
                sample_result["psnr"] = float(psnr_value)
            except Exception as e:
                print(f"PSNR計算錯誤：{str(e)}")
                sample_result["psnr"] = 0.0
            
            # 7. LPIPS計算
            try:
                # 使用修改過的 calculate_lpips_score 函數，完全支持高光譜圖像
                lpips_score = calculate_lpips_score(
                    original=images_orig[i],
                    perturbed=images_adv[i],
                    device=self.device
                )
                self.lpips_values.append(lpips_score)
                sample_result["lpips"] = float(lpips_score)
            except Exception as e:
                print(f"LPIPS計算錯誤：{str(e)}")
                sample_result["lpips"] = 0.0
                
            # 8. 光譜擾動分析
            try:
                # 進行光譜擾動分析
                spectral_impact = calculate_spectral_perturbation_impact(
                    original=images_orig[i].detach().cpu().numpy(), 
                    perturbed=images_adv[i].detach().cpu().numpy()
                )
                self.spectral_impacts.append(spectral_impact)
                
                # 從返回結果安全提取信息
                if isinstance(spectral_impact, dict) and "max_band_difference" in spectral_impact:
                    impact_data = spectral_impact
                    sample_result["spectral_impact"] = {
                        "max_band_difference": float(impact_data["max_band_difference"]),
                        "avg_band_difference": float(impact_data["avg_band_difference"]),
                        "std_band_difference": float(impact_data["std_band_difference"]),
                        "top_affected_bands": impact_data["top_affected_bands"],
                        "relative_band_impact": impact_data["relative_band_impact"],
                        "wavelength_sensitivity": float(impact_data["wavelength_sensitivity"]),
                        "band_differences": impact_data["band_differences"]
                    }
            except Exception as e:
                print(f"光譜擾動分析錯誤：{str(e)}")
                # 提供默認值以繼續執行
                sample_result["spectral_impact"] = {
                    "max_band_difference": 0.0,
                    "avg_band_difference": 0.0,
                    "std_band_difference": 0.0,
                    "top_affected_bands": [],
                    "relative_band_impact": [],
                    "wavelength_sensitivity": 0.0,
                    "band_differences": []
                }
            
            # 9. 添加分割效能指標
            # 計算原始預測和對抗預測的IoU
            pred_orig_np = pred_orig[i].detach().cpu().numpy()
            pred_adv_np = pred_adv[i].detach().cpu().numpy()
            label_np = labels[i].detach().cpu().numpy()
            
            # 添加分割指標
            segmentation_metrics = {}
            
            # 計算mIoU
            try:
                # 明確限制為二分類
                num_classes = 2
                
                # 確保預測結果在有效類別範圍內（0和1）
                pred_orig_np_valid = np.clip(pred_orig_np, 0, num_classes - 1)
                pred_adv_np_valid = np.clip(pred_adv_np, 0, num_classes - 1)
                label_np_valid = np.clip(label_np, 0, num_classes - 1)
                
                # 原始預測與標籤的IoU
                orig_iou_values = []
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_orig_np_valid != cls, label_np_valid == cls).sum()
                    union = true_pos + false_pos + false_neg
                    iou = true_pos / (union + 1e-10)
                    orig_iou_values.append(iou)
                
                # 對抗預測與標籤的IoU
                adv_iou_values = []
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_adv_np_valid != cls, label_np_valid == cls).sum()
                    union = true_pos + false_pos + false_neg
                    iou = true_pos / (union + 1e-10)
                    adv_iou_values.append(iou)
                
                # 計算平均IoU
                orig_miou = np.mean(orig_iou_values)
                adv_miou = np.mean(adv_iou_values)
                
                segmentation_metrics["original_miou"] = float(orig_miou)
                segmentation_metrics["adversarial_miou"] = float(adv_miou)
                segmentation_metrics["miou_change"] = float(adv_miou - orig_miou)
                segmentation_metrics["miou_relative_change"] = float((adv_miou - orig_miou) / (orig_miou + 1e-10))
            except Exception as e:
                print(f"mIoU計算錯誤：{str(e)}")
            
            # 計算像素分類準確率
            try:
                # 使用有效範圍內的預測結果
                # 原始預測與標籤的準確率
                orig_accuracy = np.mean(pred_orig_np_valid == label_np_valid)
                
                # 對抗預測與標籤的準確率
                adv_accuracy = np.mean(pred_adv_np_valid == label_np_valid)
                
                segmentation_metrics["original_accuracy"] = float(orig_accuracy)
                segmentation_metrics["adversarial_accuracy"] = float(adv_accuracy)
                segmentation_metrics["accuracy_change"] = float(adv_accuracy - orig_accuracy)
                segmentation_metrics["accuracy_relative_change"] = float((adv_accuracy - orig_accuracy) / (orig_accuracy + 1e-10))
            except Exception as e:
                print(f"準確率計算錯誤：{str(e)}")
            
            # 計算混淆矩陣
            try:
                # 使用已經限制好的類別數和有效範圍內的預測結果
                confusion_matrices = {}
                
                # 原始預測混淆矩陣
                orig_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                for true_class in range(num_classes):
                    for pred_class in range(num_classes):
                        orig_confusion[true_class, pred_class] = np.logical_and(
                            label_np_valid == true_class, pred_orig_np_valid == pred_class).sum()
                
                # 對抗預測混淆矩陣
                adv_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                for true_class in range(num_classes):
                    for pred_class in range(num_classes):
                        adv_confusion[true_class, pred_class] = np.logical_and(
                            label_np_valid == true_class, pred_adv_np_valid == pred_class).sum()
                
                # 預測變化混淆矩陣
                change_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                for orig_class in range(num_classes):
                    for adv_class in range(num_classes):
                        change_confusion[orig_class, adv_class] = np.logical_and(
                            pred_orig_np_valid == orig_class, pred_adv_np_valid == adv_class).sum()
                
                # 計算mIoU相關的混淆矩陣
                # 原始預測的mIoU混淆矩陣 - 真實標籤與原始預測
                miou_orig_confusion = np.zeros((num_classes, 2), dtype=np.float32)  # 每個類別的IoU計算結果
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_orig_np_valid != cls, label_np_valid == cls).sum()
                    
                    # 第一列：IoU值
                    miou_orig_confusion[cls, 0] = true_pos / (true_pos + false_pos + false_neg + 1e-10)
                    # 第二列：該類佔總樣本的百分比
                    miou_orig_confusion[cls, 1] = np.sum(label_np_valid == cls) / label_np_valid.size
                
                # 對抗樣本的mIoU混淆矩陣 - 真實標籤與對抗預測
                miou_adv_confusion = np.zeros((num_classes, 2), dtype=np.float32)
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_adv_np_valid != cls, label_np_valid == cls).sum()
                    
                    miou_adv_confusion[cls, 0] = true_pos / (true_pos + false_pos + false_neg + 1e-10)
                    miou_adv_confusion[cls, 1] = np.sum(label_np_valid == cls) / label_np_valid.size
                
                # 計算準確率相關的混淆矩陣
                # 原始預測的準確率混淆矩陣 - 每個類別的預測準確率
                acc_orig_confusion = np.zeros((num_classes, 2), dtype=np.float32)
                for cls in range(num_classes):
                    # 第一列：該類別預測正確的比例
                    true_samples = np.sum(label_np_valid == cls)
                    if true_samples > 0:
                        correct_pred = np.logical_and(pred_orig_np_valid == cls, label_np_valid == cls).sum()
                        acc_orig_confusion[cls, 0] = correct_pred / true_samples
                    else:
                        acc_orig_confusion[cls, 0] = 0.0
                    # 第二列：該類別的樣本數佔總體的比例
                    acc_orig_confusion[cls, 1] = true_samples / label_np_valid.size
                
                # 對抗樣本的準確率混淆矩陣
                acc_adv_confusion = np.zeros((num_classes, 2), dtype=np.float32)
                for cls in range(num_classes):
                    true_samples = np.sum(label_np_valid == cls)
                    if true_samples > 0:
                        correct_pred = np.logical_and(pred_adv_np_valid == cls, label_np_valid == cls).sum()
                        acc_adv_confusion[cls, 0] = correct_pred / true_samples
                    else:
                        acc_adv_confusion[cls, 0] = 0.0
                    acc_adv_confusion[cls, 1] = true_samples / label_np_valid.size
                
                # 保存所有混淆矩陣結果
                confusion_matrices["original"] = orig_confusion.tolist()
                confusion_matrices["adversarial"] = adv_confusion.tolist()
                confusion_matrices["prediction_change"] = change_confusion.tolist()
                confusion_matrices["miou_original"] = miou_orig_confusion.tolist()
                confusion_matrices["miou_adversarial"] = miou_adv_confusion.tolist()
                confusion_matrices["acc_original"] = acc_orig_confusion.tolist()
                confusion_matrices["acc_adversarial"] = acc_adv_confusion.tolist()
                
                segmentation_metrics["confusion_matrix"] = confusion_matrices
            except Exception as e:
                print(f"混淆矩陣計算錯誤：{str(e)}")
                
            # 計算類別F1分數
            try:
                # 原始預測與標籤的F1分數
                orig_f1_scores = []
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_orig_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_orig_np_valid != cls, label_np_valid == cls).sum()
                    
                    precision = true_pos / (true_pos + false_pos + 1e-10)
                    recall = true_pos / (true_pos + false_neg + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    orig_f1_scores.append(f1)
                
                # 對抗預測與標籤的F1分數
                adv_f1_scores = []
                for cls in range(num_classes):
                    true_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid == cls).sum()
                    false_pos = np.logical_and(pred_adv_np_valid == cls, label_np_valid != cls).sum()
                    false_neg = np.logical_and(pred_adv_np_valid != cls, label_np_valid == cls).sum()
                    
                    precision = true_pos / (true_pos + false_pos + 1e-10)
                    recall = true_pos / (true_pos + false_neg + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    adv_f1_scores.append(f1)
                
                # 計算平均F1分數
                orig_f1 = np.mean(orig_f1_scores)
                adv_f1 = np.mean(adv_f1_scores)
                
                segmentation_metrics["original_f1"] = float(orig_f1)
                segmentation_metrics["adversarial_f1"] = float(adv_f1)
                segmentation_metrics["f1_change"] = float(adv_f1 - orig_f1)
                segmentation_metrics["f1_relative_change"] = float((adv_f1 - orig_f1) / (orig_f1 + 1e-10))
                
                # 保存各類別的F1分數
                segmentation_metrics["original_class_f1"] = [float(f1) for f1 in orig_f1_scores]
                segmentation_metrics["adversarial_class_f1"] = [float(f1) for f1 in adv_f1_scores]
            except Exception as e:
                print(f"F1分數計算錯誤：{str(e)}")
            
            # 將分割指標添加到樣本結果中
            sample_result["segmentation_metrics"] = segmentation_metrics
            
            # 保存當前樣本的評估結果
            self.sample_metrics.append(sample_result)
    
    def get_results(self):
        """
        獲取對抗攻擊的評估結果，包含效能變化指標
        Returns:
            Dictionary containing attack metrics with performance changes
        """
        if self.total_samples == 0:
            return {
                "Attack_Success_Rate": 0.0,
                "Significant_Attack_Success_Rate": 0.0,
                "Average_L2_Norm": 0.0,
                "Average_Linf_Norm": 0.0,
                "Average_SSIM": 0.0,
                "Average_LPIPS": 0.0,
                "Performance_Changes": {},
                "Segmentation_Metrics": {}
            }
        
        # 計算每個類別的攻擊成功率
        class_success_rates = {}
        for cls, data in self.class_success_counts.items():
            if data["total"] > 0:
                class_success_rates[str(cls)] = data["count"] / data["total"]
            else:
                class_success_rates[str(cls)] = 0.0
        
        # 計算預測變化（從一個類別變為另一個類別）
        pred_changes_summary = {}
        total_pixels_changed = sum(self.pred_changes.values())
        
        if total_pixels_changed > 0:
            for change_key, count in self.pred_changes.items():
                pred_changes_summary[change_key] = count / total_pixels_changed
        
        # 計算平均光譜擾動
        spectral_summary = {
            "avg_max_band_difference": 0.0,
            "avg_top_affected_bands": []
        }
        
        if self.spectral_impacts:
            max_diffs = []
            all_top_bands = []
            
            for impact in self.spectral_impacts:
                if isinstance(impact, dict) and "max_band_difference" in impact:
                    max_diffs.append(impact["max_band_difference"])
                    
                if isinstance(impact, dict) and "top_affected_bands" in impact:
                    all_top_bands.extend(impact["top_affected_bands"])
            
            if max_diffs:
                spectral_summary["avg_max_band_difference"] = np.mean(max_diffs)
                
            if all_top_bands:
                # 找出最常見的受影響波段
                from collections import Counter
                band_counter = Counter(all_top_bands)
                most_common_bands = [band for band, _ in band_counter.most_common(10)]
                spectral_summary["avg_top_affected_bands"] = most_common_bands
        
        # 計算分割指標的變化
        # 收集所有樣本的分割指標
        orig_miou_values = []
        adv_miou_values = []
        orig_acc_values = []
        adv_acc_values = []
        
        confusion_matrices = None
        
        # 遍歷所有樣本的指標
        for sample in self.sample_metrics:
            if "segmentation_metrics" in sample:
                seg_metrics = sample["segmentation_metrics"]
                
                # 收集mIoU和準確率
                if "original_miou" in seg_metrics and "adversarial_miou" in seg_metrics:
                    orig_miou_values.append(seg_metrics["original_miou"])
                    adv_miou_values.append(seg_metrics["adversarial_miou"])
                    
                if "original_accuracy" in seg_metrics and "adversarial_accuracy" in seg_metrics:
                    orig_acc_values.append(seg_metrics["original_accuracy"])
                    adv_acc_values.append(seg_metrics["adversarial_accuracy"])
                    
                # 累積混淆矩陣
                if "confusion_matrix" in seg_metrics:
                    if confusion_matrices is None:
                        confusion_matrices = seg_metrics["confusion_matrix"]
                    else:
                        for key, matrix in seg_metrics["confusion_matrix"].items():
                            if key in confusion_matrices:
                                confusion_matrices[key] += matrix
                            else:
                                confusion_matrices[key] = matrix
        
        # 計算分割指標變化
        segmentation_metrics = {}
        
        if orig_miou_values and adv_miou_values:
            avg_orig_miou = np.mean(orig_miou_values)
            avg_adv_miou = np.mean(adv_miou_values)
            relative_miou_change = float((avg_adv_miou - avg_orig_miou) / avg_orig_miou) if avg_orig_miou > 0 else 0.0
            segmentation_metrics["Mean_IoU"] = {
                "original": float(avg_orig_miou),
                "adversarial": float(avg_adv_miou),
                "relative_change": relative_miou_change
            }
            
        if orig_acc_values and adv_acc_values:
            avg_orig_acc = np.mean(orig_acc_values)
            avg_adv_acc = np.mean(adv_acc_values)
            relative_acc_change = float((avg_adv_acc - avg_orig_acc) / avg_orig_acc) if avg_orig_acc > 0 else 0.0
            segmentation_metrics["Pixel_Accuracy"] = {
                "original": float(avg_orig_acc),
                "adversarial": float(avg_adv_acc),
                "relative_change": relative_acc_change
            }
            
        if confusion_matrices:
            # 統計類別混淆信息
            class_confusion = {}
            for key, matrix in confusion_matrices.items():
                # 確保矩陣是numpy數組，對於標準混淆矩陣限制大小為2x2
                matrix_np = np.array(matrix)
                
                # 對於標準混淆矩陣(original, adversarial, prediction_change)，限制大小為2x2
                if key in ["original", "adversarial", "prediction_change"]:
                    if matrix_np.shape[0] > 2 or matrix_np.shape[1] > 2:
                        matrix_np = matrix_np[:2, :2]
                    
                    # 計算混淆率
                    total_true = matrix_np.sum(axis=1)
                    normalized_matrix = matrix_np / (total_true[:, np.newaxis] + 1e-10)
                    class_confusion[key] = normalized_matrix.tolist()
                # 對於mIoU和acc混淆矩陣，直接保存
                elif key in ["miou_original", "miou_adversarial", "acc_original", "acc_adversarial"]:
                    # 確保只保存前兩個類別的數據
                    matrix_np = matrix_np[:2] if matrix_np.shape[0] > 2 else matrix_np
                    class_confusion[key] = matrix_np.tolist()
            
            segmentation_metrics["Class_Confusion"] = class_confusion
            
            # 添加mIoU和acc混淆矩陣的詳細資訊到結果中
            # 格式化mIoU混淆矩陣的信息
            if "miou_original" in class_confusion and "miou_adversarial" in class_confusion:
                miou_details = {
                    "original": {},
                    "adversarial": {},
                    "changes": {}
                }
                
                # 處理每個類別的mIoU
                miou_orig = np.array(class_confusion["miou_original"])
                miou_adv = np.array(class_confusion["miou_adversarial"])
                
                # 確保只處理兩個類別（二分類）
                miou_orig = miou_orig[:2] if len(miou_orig) > 2 else miou_orig
                miou_adv = miou_adv[:2] if len(miou_adv) > 2 else miou_adv
                
                for cls in range(min(2, len(miou_orig))):
                    # 原始mIoU
                    miou_details["original"][f"class_{cls}"] = float(miou_orig[cls][0])
                    # 對抗mIoU
                    miou_details["adversarial"][f"class_{cls}"] = float(miou_adv[cls][0])
                    # 變化率
                    if miou_orig[cls][0] > 0:
                        change_rate = (miou_adv[cls][0] - miou_orig[cls][0]) / miou_orig[cls][0]
                    else:
                        change_rate = 0.0
                    miou_details["changes"][f"class_{cls}"] = float(change_rate)
                
                # 平均值
                miou_details["original"]["mean"] = float(np.mean(miou_orig[:2, 0]))
                miou_details["adversarial"]["mean"] = float(np.mean(miou_adv[:2, 0]))
                if miou_details["original"]["mean"] > 0:
                    mean_change = (miou_details["adversarial"]["mean"] - miou_details["original"]["mean"]) / miou_details["original"]["mean"]
                else:
                    mean_change = 0.0
                miou_details["changes"]["mean"] = float(mean_change)
                
                segmentation_metrics["IoU_Details"] = miou_details
            
            # 格式化acc混淆矩陣的信息
            if "acc_original" in class_confusion and "acc_adversarial" in class_confusion:
                acc_details = {
                    "original": {},
                    "adversarial": {},
                    "changes": {}
                }
                
                # 處理每個類別的準確率
                acc_orig = np.array(class_confusion["acc_original"])
                acc_adv = np.array(class_confusion["acc_adversarial"])
                
                # 確保只處理兩個類別（二分類）
                acc_orig = acc_orig[:2] if len(acc_orig) > 2 else acc_orig
                acc_adv = acc_adv[:2] if len(acc_adv) > 2 else acc_adv
                
                for cls in range(min(2, len(acc_orig))):
                    # 原始準確率
                    acc_details["original"][f"class_{cls}"] = float(acc_orig[cls][0])
                    # 對抗準確率
                    acc_details["adversarial"][f"class_{cls}"] = float(acc_adv[cls][0])
                    # 變化率
                    if acc_orig[cls][0] > 0:
                        change_rate = (acc_adv[cls][0] - acc_orig[cls][0]) / acc_orig[cls][0]
                    else:
                        change_rate = 0.0
                    acc_details["changes"][f"class_{cls}"] = float(change_rate)
                
                # 平均值 (加權平均，根據每個類別的樣本比例)
                weighted_orig_acc = np.sum(acc_orig[:2, 0] * acc_orig[:2, 1])
                weighted_adv_acc = np.sum(acc_adv[:2, 0] * acc_adv[:2, 1])
                
                acc_details["original"]["weighted_mean"] = float(weighted_orig_acc)
                acc_details["adversarial"]["weighted_mean"] = float(weighted_adv_acc)
                
                if weighted_orig_acc > 0:
                    weighted_change = (weighted_adv_acc - weighted_orig_acc) / weighted_orig_acc
                else:
                    weighted_change = 0.0
                    
                acc_details["changes"]["weighted_mean"] = float(weighted_change)
                
                segmentation_metrics["Accuracy_Details"] = acc_details
        
        # 返回帶有效能變化指標和分割指標的結果
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
            
            # 4. 效能變化指標
            "Performance_Changes": {
                "Prediction_Changes": pred_changes_summary,
                "Spectral_Impact": spectral_summary
            },
            
            # 5. 分割評估指標
            "Segmentation_Metrics": segmentation_metrics
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
        self.sample_metrics = []

def calculate_lpips_score(original, perturbed, device='cpu'):
    """
    計算LPIPS (Learned Perceptual Image Patch Similarity) 分數
    
    Args:
        original (torch.Tensor): 原始圖像 [C, H, W]
        perturbed (torch.Tensor): 對抗圖像 [C, H, W]
        device (str): 設備
        
    Returns:
        float: LPIPS分數
    """
    try:
        # 確保是張量
        if not isinstance(original, torch.Tensor):
            original = torch.from_numpy(original).float()
        if not isinstance(perturbed, torch.Tensor):
            perturbed = torch.from_numpy(perturbed).float()
            
        # 移至指定設備
        original = original.to(device)
        perturbed = perturbed.to(device)
        
        # 若是批次數據，取第一個樣本
        if original.dim() == 4:
            original = original[0]
        if perturbed.dim() == 4:
            perturbed = perturbed[0]
        
        # 處理通道和空間維度
        if original.dim() != 3 or perturbed.dim() != 3:
            raise ValueError(f"輸入維度錯誤: original {original.dim()}, perturbed {perturbed.dim()}")
        
        # 使用PCA降維或選擇方差最大的通道來生成RGB圖像
        if original.size(0) > 3:
            # 使用方差最大的三個通道
            variances = torch.var(original, dim=[1, 2])
            top_channels = torch.argsort(variances, descending=True)[:3]
            
            # 創建RGB圖像
            original_rgb = torch.zeros((3, original.size(1), original.size(2)), device=device)
            perturbed_rgb = torch.zeros((3, perturbed.size(1), perturbed.size(2)), device=device)
            
            for i, channel in enumerate(top_channels):
                original_rgb[i] = original[channel]
                perturbed_rgb[i] = perturbed[channel]
                
            # 規範化到 [0, 1] 範圍
            original_rgb = (original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min() + 1e-8)
            perturbed_rgb = (perturbed_rgb - perturbed_rgb.min()) / (perturbed_rgb.max() - perturbed_rgb.min() + 1e-8)
            
            # 更新變量
            original = original_rgb
            perturbed = perturbed_rgb
        elif original.size(0) < 3:
            # 如果通道數小於3，複製通道
            original = original.repeat(3, 1, 1)[:3]
            perturbed = perturbed.repeat(3, 1, 1)[:3]
        
        # 確保空間尺寸至少為64x64（LPIPS推薦最小尺寸）
        if original.size(1) < 64 or original.size(2) < 64:
            original = F.interpolate(original.unsqueeze(0), size=(max(64, original.size(1)), max(64, original.size(2))), mode='bilinear').squeeze(0)
        if perturbed.size(1) < 64 or perturbed.size(2) < 64:
            perturbed = F.interpolate(perturbed.unsqueeze(0), size=(max(64, perturbed.size(1)), max(64, perturbed.size(2))), mode='bilinear').squeeze(0)
        
        # 確保數值範圍在[-1, 1]，LPIPS期望的是[-1, 1]範圍
        if original.min() >= 0 and original.max() <= 1:
            original = original * 2 - 1
        if perturbed.min() >= 0 and perturbed.max() <= 1:
            perturbed = perturbed * 2 - 1
        
        # 添加批次維度
        original = original.unsqueeze(0)
        perturbed = perturbed.unsqueeze(0)
        
        # 獲取LPIPS模型
        lpips_fn = get_lpips_model(device)
        
        # 計算LPIPS分數
        with torch.no_grad():
            distance = lpips_fn(original, perturbed)
            
        return distance.item()
    except Exception as e:
        print(f"LPIPS計算錯誤：{str(e)}")
        return 0.0  # 返回默認值