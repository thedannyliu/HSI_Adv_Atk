import torch
import torch.nn.functional as F
from .attack_utils import normalize, unnormalize
from .attack_utils import perceptual_constraint, adaptive_perturbation_size
from .attack_utils import spectral_importance_analysis, selective_perturbation

def fgsm_attack_adaptive(model, images, labels, eps, criterion, device, mean=None, std=None,
                        apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                        use_adaptive_eps=False, min_eps=0.01, max_eps=0.1):
    """
    自適應快速梯度符號法(FGSM)攻擊
    
    通過計算損失函數關於輸入的梯度，沿著梯度符號方向進行擾動，生成對抗樣本。
    該版本支持自適應調整擾動方向以便更好地攻擊多樣化的高光譜圖像。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值，可選
        std (list or float): 正規化標準差，可選
        apply_perceptual_constraint (bool): 是否應用感知限制
        ssim_threshold (float): SSIM閾值，高於此值視為可接受
        lpips_threshold (float): LPIPS閾值，低於此值視為可接受
        use_adaptive_eps (bool): 是否使用自適應擾動大小
        min_eps (float): 最小擾動大小
        max_eps (float): 最大擾動大小
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 添加數據範圍自適應的擾動放大因子
    # 高光譜數據通常範圍很大，需要更大的擾動才能產生明顯效果
    data_range_factor = 10.0  # 基本放大因子
    
    # 根據數據估算合適的放大因子
    with torch.no_grad():
        orig_images_unnorm = unnormalize(images.clone(), mean, std)
        data_range = torch.max(orig_images_unnorm) - torch.min(orig_images_unnorm)
        if data_range > 1.0:  # 高光譜數據通常範圍較大
            # 根據數據範圍動態調整放大因子，最小10倍，最大50倍
            data_range_factor = max(min(data_range / 10.0, 50.0), 10.0)
    
    print(f"使用數據範圍自適應放大因子: {data_range_factor:.2f}, 數據範圍: {data_range:.2f}")
    
    # 應用數據範圍因子到擾動上
    effective_eps = eps * data_range_factor
    
    # 計算自適應擾動大小（如果需要）
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device).requires_grad_(True)
    
    # 前向傳播
    outputs = model(images_norm)
    
    # 處理可能的多輸出情況
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # 計算損失
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()
    
    # 提取梯度
    grad = images_norm.grad.data
    
    # 自適應調整：計算每個通道的梯度重要性
    channel_importance = torch.mean(grad.abs(), dim=(2, 3), keepdim=True)
    
    # 歸一化以保持總擾動大小不變
    channel_importance = channel_importance / (torch.sum(channel_importance, dim=1, keepdim=True) + 1e-8)
    
    # 應用通道重要性到梯度
    weighted_grad = grad * channel_importance
    grad_sign = torch.sign(weighted_grad)
    
    # 計算擾動（在正規化空間）
    if use_adaptive_eps:
        perturbation = adaptive_eps * grad_sign
    else:
        perturbation = effective_eps * grad_sign
    
    # 添加擾動到正規化圖像
    adv_images_norm = images_norm + perturbation
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images_norm = perceptual_constraint(
            images, adv_images_norm, ssim_threshold, lpips_threshold, device
        )
    
    # 計算原始域的擾動大小
    with torch.no_grad():
        images_unnorm = unnormalize(images, mean, std)
        adv_images_unnorm = unnormalize(adv_images_norm, mean, std)
        actual_perturbation = adv_images_unnorm - images_unnorm
        actual_perturbation_l_inf = torch.norm(actual_perturbation.view(images.size(0), -1), p=float('inf'), dim=1)
        actual_perturbation_l2 = torch.norm(actual_perturbation.view(images.size(0), -1), p=2, dim=1)
    
    # 輸出調試信息
    B = images.size(0)
    perturbation_l_inf = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
    print(f"基本FGSM: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"基本FGSM: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"基本FGSM: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    
    # 測試攻擊效果
    with torch.no_grad():
        orig_output = model(images)
        adv_output = model(adv_images_norm)
        if isinstance(orig_output, tuple):
            orig_output = orig_output[0]
        if isinstance(adv_output, tuple):
            adv_output = adv_output[0]
        orig_pred = orig_output.argmax(1)
        adv_pred = adv_output.argmax(1)
        success_rate = (orig_pred != adv_pred).float().mean()
        print(f"基本FGSM: 攻擊成功率: {success_rate.item():.4f}")
    
    return adv_images_norm.detach()

def pgd_attack(model, images, labels, eps, alpha, steps, criterion, device, mean=None, std=None,
              apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
              use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
              use_spectral_importance=False, spectral_threshold=0.7):
    """
    投影梯度下降(PGD)攻擊
    
    通過多步迭代優化，沿著梯度方向進行擾動，通過投影確保擾動在範圍內，生成高質量對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        eps (float): 最大擾動範圍
        alpha (float): 步長
        steps (int): 迭代次數
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值，可選
        std (list or float): 正規化標準差，可選
        apply_perceptual_constraint (bool): 是否應用感知限制
        ssim_threshold (float): SSIM閾值，高於此值視為可接受
        lpips_threshold (float): LPIPS閾值，低於此值視為可接受
        use_adaptive_eps (bool): 是否使用自適應擾動大小
        min_eps (float): 最小擾動大小
        max_eps (float): 最大擾動大小
        use_spectral_importance (bool): 是否使用光譜重要性分析
        spectral_threshold (float): 光譜重要性閾值
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 添加數據範圍自適應的擾動放大因子
    # 高光譜數據通常範圍很大，需要更大的擾動才能產生明顯效果
    data_range_factor = 10.0  # 基本放大因子
    
    # 根據數據估算合適的放大因子
    with torch.no_grad():
        if mean is not None and std is not None:
            orig_images_unnorm = unnormalize(images.clone(), mean, std)
            data_range = torch.max(orig_images_unnorm) - torch.min(orig_images_unnorm)
            if data_range > 1.0:  # 高光譜數據通常範圍較大
                # 根據數據範圍動態調整放大因子，最小10倍，最大50倍
                data_range_factor = max(min(data_range / 10.0, 50.0), 10.0)
    
    print(f"基本PGD: 使用數據範圍自適應放大因子: {data_range_factor:.2f}")
    
    # 應用數據範圍因子到擾動上
    effective_eps = eps * data_range_factor
    effective_alpha = alpha * data_range_factor
    
    # 如果使用自適應擾動大小，也需要放大
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
    
    # 直接在正規化域中操作
    # 如果有提供mean和std，確保圖像已經正規化
    if mean is not None and std is not None:
        images_norm = normalize(images.clone(), mean, std).to(device)
    else:
        images_norm = images.clone().detach().to(device)
    
    adv_images_norm = images_norm.clone().detach()
    
    B, C, H, W = images_norm.shape
    
    # 如果使用光譜重要性分析，先計算重要波段
    if use_spectral_importance:
        band_importance = spectral_importance_analysis(images, labels, model, device)
        important_bands = []
        
        # 選擇重要性最高的波段直到達到閾值
        sorted_bands = torch.argsort(band_importance, descending=True)
        cumulative_importance = 0
        for band_idx in sorted_bands:
            important_bands.append(band_idx.item())
            cumulative_importance += band_importance[band_idx]
            if cumulative_importance >= spectral_threshold:
                break
        print(f"基本PGD: 選擇了 {len(important_bands)}/{C} 個波段，累積重要性: {cumulative_importance:.4f}")
    
    for i in range(steps):
        adv_images_norm.requires_grad = True
        
        # 前向傳播
        outputs = model(adv_images_norm)
        
        # 處理可能的多輸出情況
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算損失
        loss = criterion(outputs, labels)
        
        # 反向傳播
        model.zero_grad()
        loss.backward()
        
        # 提取梯度
        grad = adv_images_norm.grad.data
        grad_sign = torch.sign(grad)
        
        # 應用光譜重要性分析結果
        if use_spectral_importance:
            # 選擇性地應用擾動（只擾動重要波段）
            spectral_mask = torch.zeros((B, C, 1, 1), device=device)
            for band_idx in important_bands:
                spectral_mask[:, band_idx, 0, 0] = 1.0
            masked_grad_sign = grad_sign * spectral_mask.expand_as(grad_sign)
            perturbation = effective_alpha * masked_grad_sign
        else:
            perturbation = effective_alpha * grad_sign
        
        # 添加擾動
        adv_images_norm = adv_images_norm.detach() + perturbation
        
        # 計算與原始圖像的差距
        if use_adaptive_eps:
            # 為每個樣本單獨應用擾動限制
            for b in range(B):
                eta = adv_images_norm[b] - images_norm[b]
                eta = torch.clamp(eta, -adaptive_eps[b, 0, 0, 0].item(), adaptive_eps[b, 0, 0, 0].item())
                adv_images_norm[b] = images_norm[b] + eta
        else:
            # 使用固定的epsilon
            eta = adv_images_norm - images_norm
            eta = torch.clamp(eta, -effective_eps, effective_eps)
            adv_images_norm = images_norm + eta
    
    # 計算原始域的擾動大小
    with torch.no_grad():
        if mean is not None and std is not None:
            images_unnorm = unnormalize(images, mean, std)
            adv_images_unnorm = unnormalize(adv_images_norm, mean, std)
            actual_perturbation = adv_images_unnorm - images_unnorm
            actual_perturbation_l_inf = torch.norm(actual_perturbation.view(B, -1), p=float('inf'), dim=1)
            actual_perturbation_l2 = torch.norm(actual_perturbation.view(B, -1), p=2, dim=1)
            
            print(f"基本PGD: 正規化域平均L∞擾動大小: {torch.norm((adv_images_norm - images_norm).view(B, -1), p=float('inf'), dim=1).mean().item():.4f}")
            print(f"基本PGD: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
            print(f"基本PGD: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
            
            # 測試攻擊效果
            orig_output = model(images_norm)
            adv_output = model(adv_images_norm)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            if isinstance(adv_output, tuple):
                adv_output = adv_output[0]
            orig_pred = orig_output.argmax(1)
            adv_pred = adv_output.argmax(1)
            success_rate = (orig_pred != adv_pred).float().mean()
            print(f"基本PGD: 攻擊成功率: {success_rate.item():.4f}")
            
            # 反正規化最終結果
            adv_images = adv_images_unnorm
        else:
            # 如果沒有提供正規化參數，直接返回正規化後的結果
            adv_images = adv_images_norm
            
            # 輸出基本擾動信息
            perturbation_l_inf = torch.norm((adv_images_norm - images_norm).view(B, -1), p=float('inf'), dim=1)
            perturbation_l2 = torch.norm((adv_images_norm - images_norm).view(B, -1), p=2, dim=1)
            print(f"基本PGD: 平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
            print(f"基本PGD: 平均L2擾動大小: {perturbation_l2.mean().item():.4f}")
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images = perceptual_constraint(
            images, adv_images, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images.detach()