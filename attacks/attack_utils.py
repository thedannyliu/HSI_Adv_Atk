import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import warnings

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

def normalize(x, mean=None, std=None):
    """
    正規化輸入數據
    Args:
        x (torch.Tensor): 輸入張量 [B, C, H, W]
        mean (float, list, torch.Tensor): 均值
        std (float, list, torch.Tensor): 標準差
    Returns:
        torch.Tensor: 正規化後的張量
    """
    if mean is None:
        mean = 0.0
    if std is None:
        std = 1.0
        
    # 處理不同類型的mean和std
    if isinstance(mean, (int, float)):
        mean = torch.tensor([mean] * x.size(1)).view(1, -1, 1, 1).to(x.device)
    elif isinstance(mean, list):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(x.device)
    
    if isinstance(std, (int, float)):
        std = torch.tensor([std] * x.size(1)).view(1, -1, 1, 1).to(x.device)
    elif isinstance(std, list):
        std = torch.tensor(std).view(1, -1, 1, 1).to(x.device)
        
    return (x - mean) / std

def unnormalize(x, mean=None, std=None):
    """
    反正規化處理
    Args:
        x (torch.Tensor): 輸入張量 [B, C, H, W]
        mean (float, list, torch.Tensor): 均值
        std (float, list, torch.Tensor): 標準差
    Returns:
        torch.Tensor: 反正規化後的張量
    """
    if mean is None:
        mean = 0.0
    if std is None:
        std = 1.0
        
    # 處理不同類型的mean和std
    if isinstance(mean, (int, float)):
        mean = torch.tensor([mean] * x.size(1)).view(1, -1, 1, 1).to(x.device)
    elif isinstance(mean, list):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(x.device)
    
    if isinstance(std, (int, float)):
        std = torch.tensor([std] * x.size(1)).view(1, -1, 1, 1).to(x.device)
    elif isinstance(std, list):
        std = torch.tensor(std).view(1, -1, 1, 1).to(x.device)
        
    return x * std + mean

def calculate_ssim(original, perturbed, data_range=1.0):
    """
    計算SSIM (結構相似性)
    Args:
        original (torch.Tensor): 原始圖像
        perturbed (torch.Tensor): 擾動圖像
        data_range (float): 數據範圍
    Returns:
        float: SSIM值
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(perturbed, torch.Tensor):
        perturbed = perturbed.detach().cpu().numpy()
        
    try:
        # 添加保護，確保圖像不完全相同
        if np.array_equal(original, perturbed):
            return 1.0  # 完全相同返回1.0
            
        # 確保數據範圍合理
        if data_range <= 0 or np.isnan(data_range) or np.isinf(data_range):
            computed_range = np.max(original) - np.min(original)
            data_range = max(computed_range, 1e-8)
            
        # 確保數據有效
        if np.isnan(original).any() or np.isnan(perturbed).any() or \
           np.isinf(original).any() or np.isinf(perturbed).any():
            print("警告: 發現NaN或Inf值，SSIM計算可能不準確")
            # 替換無效值
            original = np.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0)
            perturbed = np.nan_to_num(perturbed, nan=0.0, posinf=1.0, neginf=0.0)
            
        return ssim(original, perturbed, channel_axis=0, data_range=data_range)
    except Exception as e:
        print(f"SSIM計算錯誤: {str(e)}")
        return 0.5  # 返回中間值作為默認

def calculate_lpips(original, perturbed, device='cpu'):
    """
    計算LPIPS (學習感知圖像相似度)
    Args:
        original (torch.Tensor): 原始圖像 [C, H, W]
        perturbed (torch.Tensor): 擾動圖像 [C, H, W]
        device (str): 計算設備
    Returns:
        float: LPIPS值 (值越大表示差異越大)
    """
    try:
        # 獲取LPIPS模型
        model = get_lpips_model(device)
        
        # 確保是張量並且是浮點類型
        if not isinstance(original, torch.Tensor):
            original = torch.from_numpy(original).float().to(device)
        else:
            original = original.float().to(device)
            
        if not isinstance(perturbed, torch.Tensor):
            perturbed = torch.from_numpy(perturbed).float().to(device)
        else:
            perturbed = perturbed.float().to(device)
        
        # 檢查輸入數據是否有NaN或Inf
        if torch.isnan(original).any() or torch.isnan(perturbed).any() or \
           torch.isinf(original).any() or torch.isinf(perturbed).any():
            print("警告: 發現NaN或Inf值，LPIPS計算可能不準確")
            # 替換無效值
            original = torch.nan_to_num(original, nan=0.0, posinf=1.0, neginf=0.0)
            perturbed = torch.nan_to_num(perturbed, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 檢查是否完全相同
        if torch.allclose(original, perturbed, rtol=1e-5, atol=1e-8):
            return 0.0  # 完全相同返回0.0 (最小差異)
        
        # 調整到LPIPS期望的格式 [1, 3, H, W]
        if original.dim() == 3:
            # 如果有超過3個通道，選擇最有代表性的3個通道
            if original.size(0) >= 3:
                # 使用方差最大的3個通道 - 確保輸入是浮點型
                original_float = original.float()  # 確保是浮點型
                
                # 強制轉換為浮點類型並計算方差
                variances = torch.var(original_float.float(), dim=(1, 2))
                
                # 檢查是否計算成功
                if torch.isnan(variances).any():
                    # 如果有NaN值，則使用前三個通道
                    indices = torch.tensor([0, 1, 2], device=device)
                else:
                    # 獲取方差最大的三個通道
                    _, indices = torch.topk(variances, min(3, len(variances)))
                
                # 創建RGB張量
                orig_rgb = torch.zeros(3, original.size(1), original.size(2), device=device, dtype=torch.float32)
                pert_rgb = torch.zeros(3, perturbed.size(1), perturbed.size(2), device=device, dtype=torch.float32)
                
                # 確保indices的數量不超過3
                for j in range(min(len(indices), 3)):
                    # 確保索引在有效範圍內
                    idx = indices[j] if j < len(indices) else 0
                    ch_idx = min(j, 2)  # 確保通道索引不超過2
                    orig_rgb[ch_idx] = original[idx].float()
                    pert_rgb[ch_idx] = perturbed[idx].float()
                
                # 確保數據範圍在[0,1]
                orig_rgb = torch.clamp(orig_rgb, 0, 1)
                pert_rgb = torch.clamp(pert_rgb, 0, 1)
                
                # 添加批次維度
                orig_rgb = orig_rgb.unsqueeze(0)
                pert_rgb = pert_rgb.unsqueeze(0)
            else:
                # 如果通道數少於3，複製當前通道
                orig_rgb = original.float().repeat(3, 1, 1) if original.size(0) == 1 else original.float()
                pert_rgb = perturbed.float().repeat(3, 1, 1) if perturbed.size(0) == 1 else perturbed.float()
                
                # 確保三通道
                if orig_rgb.size(0) < 3:
                    # 複製最後一個通道來填充
                    last_channel = orig_rgb[-1:].clone()
                    while orig_rgb.size(0) < 3:
                        orig_rgb = torch.cat([orig_rgb, last_channel], dim=0)
                    
                    last_channel = pert_rgb[-1:].clone()
                    while pert_rgb.size(0) < 3:
                        pert_rgb = torch.cat([pert_rgb, last_channel], dim=0)
                
                # 確保數據範圍在[0,1]
                orig_rgb = torch.clamp(orig_rgb, 0, 1)
                pert_rgb = torch.clamp(pert_rgb, 0, 1)
                
                # 添加批次維度
                orig_rgb = orig_rgb.unsqueeze(0)
                pert_rgb = pert_rgb.unsqueeze(0)
        else:
            # 處理非3維張量的情況
            print(f"Warning: 非預期的張量維度 {original.dim()}，LPIPS需要[C, H, W]格式")
            # 創建一個簡單的3通道圖像
            orig_rgb = torch.zeros((1, 3, 64, 64), device=device, dtype=torch.float32)
            pert_rgb = torch.zeros((1, 3, 64, 64), device=device, dtype=torch.float32)
        
        # 確保輸入到 LPIPS 模型的是浮點數類型
        orig_rgb = orig_rgb.float()
        pert_rgb = pert_rgb.float()
        
        # 檢查形狀是否符合要求
        if orig_rgb.dim() != 4 or orig_rgb.size(1) != 3:
            raise ValueError(f"LPIPS輸入形狀錯誤: {orig_rgb.shape}，期望[1, 3, H, W]")
        
        # 計算LPIPS值
        with torch.no_grad():
            lpips_value = model(orig_rgb, pert_rgb).item()
        
        # 確保返回值是有效數字
        if np.isnan(lpips_value) or np.isinf(lpips_value):
            return 0.05  # 返回一個合理的默認值
            
        return lpips_value
    except Exception as e:
        print(f"LPIPS計算錯誤：{str(e)}")
        return 0.05  # 在錯誤情況下返回默認值

def perceptual_constraint(original, perturbed, ssim_threshold=0.95, lpips_threshold=0.05, device='cpu'):
    """
    實現感知限制，確保擾動在視覺上不可見
    
    Args:
        original (torch.Tensor): 原始圖像 [B, C, H, W]
        perturbed (torch.Tensor): 擾動後圖像 [B, C, H, W]
        ssim_threshold (float): SSIM閾值，高於此值視為可接受
        lpips_threshold (float): LPIPS閾值，低於此值視為可接受
        device (str): 計算設備
        
    Returns:
        torch.Tensor: 調整後的對抗樣本
    """
    adjusted_samples = []
    
    for i in range(original.size(0)):
        orig_img = original[i]
        pert_img = perturbed[i]
        
        # 計算當前SSIM
        current_ssim = calculate_ssim(orig_img.cpu(), pert_img.cpu())
        
        # 計算當前LPIPS
        current_lpips = calculate_lpips(orig_img, pert_img, device)
        
        # 若擾動過大，進行調整
        if current_ssim < ssim_threshold or current_lpips > lpips_threshold:
            # 計算擾動
            perturbation = pert_img - orig_img
            
            # 根據SSIM和LPIPS差距計算調整係數
            if current_ssim < ssim_threshold:
                ssim_factor = current_ssim / ssim_threshold
            else:
                ssim_factor = 1.0
                
            if current_lpips > lpips_threshold:
                lpips_factor = lpips_threshold / (current_lpips + 1e-8)  # 避免除零
            else:
                lpips_factor = 1.0
            
            # 取較小的縮放因子
            scale_factor = min(ssim_factor, lpips_factor)
            
            # 確保縮放因子有效且合理
            scale_factor = max(min(scale_factor, 1.0), 0.1)  # 限制在0.1到1.0之間
            
            # 調整擾動幅度
            adjusted_perturbation = perturbation * scale_factor
            
            # 生成新的對抗樣本
            adjusted_img = orig_img + adjusted_perturbation
            
            # 檢查調整後的樣本是否包含無效值
            if torch.isnan(adjusted_img).any() or torch.isinf(adjusted_img).any():
                print(f"警告: 樣本 {i} 調整後包含NaN或Inf值，使用原始對抗樣本")
                adjusted_samples.append(pert_img)
            else:
                adjusted_samples.append(adjusted_img)
            
            print(f"樣本 {i}: SSIM {current_ssim:.4f} -> {calculate_ssim(orig_img.cpu(), adjusted_img.cpu()):.4f}, "
                  f"LPIPS {current_lpips:.4f} -> {calculate_lpips(orig_img, adjusted_img, device):.4f}")
        else:
            # 如果已經滿足感知約束，則不調整
            adjusted_samples.append(pert_img)
    
    # 合併批次
    return torch.stack(adjusted_samples).to(device)

def adaptive_perturbation_size(images, labels, model, base_eps=0.03, min_eps=0.01, max_eps=0.1, num_steps=5):
    """
    根據圖像的不同特性計算每個樣本的自適應擾動大小
    
    Args:
        images (torch.Tensor): 輸入圖像 [B, C, H, W]
        labels (torch.Tensor): 真實標籤 [B, H, W]
        model (nn.Module): 目標模型
        base_eps (float): 基準擾動大小
        min_eps (float): 最小擾動大小
        max_eps (float): 最大擾動大小
        num_steps (int): 嘗試不同擾動大小的步數
        
    Returns:
        torch.Tensor: 每個樣本的自適應擾動大小 [B, 1, 1, 1]
    """
    B = images.size(0)
    device = images.device
    
    # 初始化為最小擾動
    adaptive_eps = torch.ones(B, 1, 1, 1, device=device) * min_eps
    
    # 獲取初始預測
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        orig_preds = outputs.argmax(1)
    
    # 創建擾動大小列表
    eps_values = torch.linspace(min_eps, max_eps, num_steps).to(device)
    
    # 對每個樣本測試不同的擾動大小
    for i in range(B):
        found_success = False
        
        # 獲取當前樣本
        img = images[i:i+1]
        label = labels[i:i+1]
        orig_pred = orig_preds[i:i+1]
        
        # 嘗試不同擾動大小
        for eps in eps_values:
            # 創建隨機噪聲
            noise = (torch.rand_like(img) * 2 - 1) * eps
            
            # 添加噪聲但不限制到0-1範圍
            adv_img = img + noise
            
            # 獲取對抗預測
            with torch.no_grad():
                outputs = model(adv_img)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                adv_pred = outputs.argmax(1)
            
            # 檢查是否攻擊成功
            if (adv_pred != orig_pred).any():
                adaptive_eps[i] = eps
                found_success = True
                break
        
        # 如果沒有找到合適的擾動，使用最大值
        if not found_success:
            adaptive_eps[i] = max_eps
    
    return adaptive_eps

def spectral_importance_analysis(images, labels, model, device):
    """
    分析波段重要性，用於選擇性擾動
    
    Args:
        images (torch.Tensor): 輸入圖像 [B, C, H, W]
        labels (torch.Tensor): 真實標籤 [B, H, W]
        model (nn.Module): 目標模型
        device (str): 計算設備
        
    Returns:
        torch.Tensor: 波段重要性分數 [C]
    """
    band_importance = []
    num_bands = images.size(1)
    
    # 獲取原始預測
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        original_preds = outputs.argmax(dim=1)
    
    # 對每個波段單獨添加小擾動，分析影響
    for band_idx in range(num_bands):
        changes_sum = 0
        
        # 生成擾動
        perturbed = images.clone()
        noise = torch.zeros_like(perturbed)
        noise[:, band_idx:band_idx+1] = 0.05 * torch.randn_like(perturbed[:, band_idx:band_idx+1])
        perturbed = perturbed + noise
        
        # 預測擾動樣本
        with torch.no_grad():
            outputs = model(perturbed)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            perturbed_preds = outputs.argmax(dim=1)
        
        # 計算變化比例
        for i in range(images.size(0)):
            # 只考慮有目標區域的像素
            target_mask = (labels[i] > 0)
            if target_mask.sum() > 0:
                # 計算預測變化率 - 修復可能的索引問題
                if target_mask.bool().any():  # 確保掩碼中有 True 值
                    pred_changed = (original_preds[i] != perturbed_preds[i]) & target_mask.bool()
                    change_rate = pred_changed.float().sum() / target_mask.float().sum()
                    changes_sum += change_rate.item()
            # 如果沒有目標區域，不添加變化
        
        # 平均變化率作為重要性
        importance = changes_sum / images.size(0) if images.size(0) > 0 else 0
        band_importance.append(importance)
    
    # 轉換為張量
    band_importance = torch.tensor(band_importance, device=device)
    
    # 歸一化
    if band_importance.sum() > 0:
        band_importance = band_importance / band_importance.sum()
    
    return band_importance

def selective_perturbation(perturbation, band_importance, threshold=0.7):
    """
    根據波段重要性選擇性地應用擾動
    
    Args:
        perturbation (torch.Tensor): 原始擾動 [B, C, H, W]
        band_importance (torch.Tensor): 波段重要性分數 [C]
        threshold (float): 重要性閾值，超過此值的波段將被選擇
        
    Returns:
        torch.Tensor: 選擇性擾動 [B, C, H, W]
    """
    # 計算累積重要性
    sorted_importance, indices = torch.sort(band_importance, descending=True)
    cumulative_importance = torch.cumsum(sorted_importance, dim=0)
    
    # 根據閾值選擇波段
    mask_threshold = (cumulative_importance <= threshold).float()
    
    # 創建重要波段遮罩
    important_bands = torch.zeros_like(band_importance)
    important_bands[indices[mask_threshold.bool()]] = 1.0
    
    # 擴展為與擾動相同的形狀
    band_mask = important_bands.view(1, -1, 1, 1).expand_as(perturbation)
    
    # 應用選擇性擾動
    selective_pert = perturbation * band_mask
    
    return selective_pert