import torch
import torch.nn.functional as F
from .utils import zero_gradients, get_important_bands
from .attack_utils import normalize, unnormalize
from .attack_utils import perceptual_constraint, adaptive_perturbation_size
from .attack_utils import spectral_importance_analysis, selective_perturbation

def fgsm_spectral_attack(model, images, labels, eps, criterion, device, mean, std, target_bands=None,
                        apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                        use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                        use_spectral_importance=False, spectral_threshold=0.9,
                        **kwargs):
    """
    光譜域上的快速梯度符號法(FGSM)攻擊
    
    專注於修改特定光譜波段，保持空間結構不變，實現更隱蔽的對抗樣本生成。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像，形狀為 [B, C, H, W]
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        target_bands (list): 要攻擊的目標波段，如果為None，會自動選擇重要波段
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
        orig_images_unnorm = unnormalize(images.clone(), mean, std)
        data_range = torch.max(orig_images_unnorm) - torch.min(orig_images_unnorm)
        if data_range > 1.0:  # 高光譜數據通常範圍較大
            # 根據數據範圍動態調整放大因子，最小10倍，最大50倍
            data_range_factor = max(min(data_range / 10.0, 50.0), 10.0)
    
    print(f"使用數據範圍自適應放大因子: {data_range_factor:.2f}, 數據範圍: {data_range:.2f}")
    
    # 應用數據範圍因子到擾動上
    effective_eps = eps * data_range_factor
    
    # 如果使用自適應擾動大小，也需要放大
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device).requires_grad_(True)
    
    B, C, H, W = images_norm.shape
    
    # 如果使用光譜重要性分析，優先使用分析結果
    if use_spectral_importance:
        band_importance = spectral_importance_analysis(images, labels, model, device)
        target_bands = []
        
        # 選擇重要性最高的波段直到達到閾值
        sorted_bands = torch.argsort(band_importance, descending=True)
        cumulative_importance = 0
        for band_idx in sorted_bands:
            target_bands.append(band_idx.item())
            cumulative_importance += band_importance[band_idx]
            if cumulative_importance >= spectral_threshold:
                break
        
        # 確保至少選擇30%的波段
        if len(target_bands) < C * 0.3:
            additional_bands = sorted_bands[len(target_bands):int(C * 0.3)]
            target_bands.extend(additional_bands.tolist())
            
    # 如果沒有指定目標波段，使用所有通道
    elif target_bands is None:
        target_bands = list(range(C))
    
    # Forward pass
    outputs = model(images_norm)
    if isinstance(outputs, tuple):
        # 若模型回傳 (pred, pred_bands)，取 pred 用來計算 loss
        outputs = outputs[0]
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()

    # 取得梯度
    grad = images_norm.grad.data
    
    # 計算每個波段的重要性
    band_grad_importance = torch.mean(torch.abs(grad), dim=(0, 2, 3))
    
    # 創建光譜掩碼 - 使用正確的維度 [B, C, 1, 1]
    spectral_mask = torch.zeros((B, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[:, band_idx, 0, 0] = 1.0
    
    # 使用符號梯度和掩碼生成擾動
    grad_sign = torch.sign(grad)
    
    # 應用掩碼
    masked_grad_sign = grad_sign * spectral_mask
    
    # 使用自適應擾動大小或固定擾動大小
    if use_adaptive_eps:
        perturbation = adaptive_eps * masked_grad_sign
    else:
        perturbation = effective_eps * masked_grad_sign
    
    # 生成對抗樣本
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
        actual_perturbation_l_inf = torch.norm(actual_perturbation.view(B, -1), p=float('inf'), dim=1)
        actual_perturbation_l2 = torch.norm(actual_perturbation.view(B, -1), p=2, dim=1)
    
    # 輸出調試信息
    perturbation_l_inf = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
    print(f"光譜FGSM: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"光譜FGSM: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"光譜FGSM: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    print(f"光譜FGSM: 選擇的波段數量: {len(target_bands)}/{C}")
    
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
        print(f"光譜FGSM: 攻擊成功率: {success_rate.item():.4f}")
    
    return adv_images_norm.detach()


def pgd_spectral_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std, target_bands=None,
                       apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                       use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                       use_spectral_importance=False, spectral_threshold=0.7,
                       **kwargs):
    """
    光譜域上的投影梯度下降(PGD)攻擊
    
    通過多步優化，針對特定光譜波段生成對抗樣本，實現更強效的光譜擾動。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        eps (float): 最大擾動範圍
        alpha (float): 步長
        steps (int): 迭代次數
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        target_bands (list): 要攻擊的目標波段，如果為None，會自動選擇重要波段
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
    # 計算自適應擾動大小
    if use_adaptive_eps:
        adaptive_eps = adaptive_perturbation_size(images, labels, model, eps, min_eps, max_eps)
    
    # 反正規化
    images_orig = unnormalize(images, mean, std).clone().detach().to(device)
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    adv_images_unnorm = images_orig.clone().detach()
    
    B, C, H, W = images_orig.shape
    
    # 如果使用光譜重要性分析，優先使用分析結果
    if use_spectral_importance:
        band_importance = spectral_importance_analysis(images, labels, model, device)
        target_bands = []
        
        # 選擇重要性最高的波段直到達到閾值
        sorted_bands = torch.argsort(band_importance, descending=True)
        cumulative_importance = 0
        for band_idx in sorted_bands:
            target_bands.append(band_idx.item())
            cumulative_importance += band_importance[band_idx]
            if cumulative_importance >= spectral_threshold:
                break
    # 如果沒有指定目標波段，使用所有通道
    elif target_bands is None:
        target_bands = list(range(C))
    
    # 創建光譜掩碼
    spectral_mask = torch.zeros((B, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[:, band_idx, 0, 0] = 1.0
    
    for i in range(steps):
        # 創建新的葉節點變數而不是修改現有變數的 requires_grad
        adv_images_temp = adv_images_unnorm.clone().detach().requires_grad_(True)
        
        outputs = model(normalize(adv_images_temp, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_images_temp.grad.data
        
        # 應用光譜掩碼
        masked_grad = grad * spectral_mask
        grad_sign = masked_grad.sign()
        
        # 更新對抗樣本
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        
        # 確保擾動在限制範圍內
        if use_adaptive_eps:
            # 根據每個樣本的自適應擾動大小計算限制
            eta = torch.zeros_like(adv_images_unnorm)
            for b in range(B):
                sample_eps = adaptive_eps[b].item() if hasattr(adaptive_eps[b], 'item') else adaptive_eps[b]
                eta[b] = torch.clamp(adv_images_unnorm[b] - images_orig[b], min=-sample_eps, max=sample_eps)
        else:
            eta = torch.clamp(adv_images_unnorm - images_orig, min=-eps, max=eps)
            
        adv_images_unnorm = images_orig + eta
        # 不再限制到0-1範圍，讓圖像保持在原始數據範圍
    
    adv_images = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images = perceptual_constraint(
            images, adv_images, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images.detach()


def cw_spectral_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1,
                     device='cuda', mean=None, std=None, target_bands=None,
                     apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                     use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                     use_spectral_importance=False, spectral_threshold=0.7,
                     **kwargs):
    """
    光譜域Carlini & Wagner攻擊
    
    通過優化L2範數最小的擾動，專注於光譜波段，實現高質量的對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        c (float): 平衡係數
        kappa (float): 置信度參數
        steps (int): 優化步數
        lr (float): 學習率
        eps (float): 最大擾動範圍
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        target_bands (list): 要攻擊的目標波段，如果為None，會自動選擇重要波段
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
    # 計算自適應擾動大小
    if use_adaptive_eps:
        adaptive_eps_tensor = adaptive_perturbation_size(images, labels, model, eps, min_eps, max_eps)
        eps_to_use = adaptive_eps_tensor.view(-1)[0].item()  # 使用第一個樣本的擾動大小
    else:
        eps_to_use = eps
    
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    B, C, H, W = images_unnorm.shape
    
    # 如果使用光譜重要性分析，優先使用分析結果
    if use_spectral_importance:
        band_importance = spectral_importance_analysis(images, labels, model, device)
        target_bands = []
        
        # 選擇重要性最高的波段直到達到閾值
        sorted_bands = torch.argsort(band_importance, descending=True)
        cumulative_importance = 0
        for band_idx in sorted_bands:
            target_bands.append(band_idx.item())
            cumulative_importance += band_importance[band_idx]
            if cumulative_importance >= spectral_threshold:
                break
    # 如果沒有指定目標波段，使用所有通道
    elif target_bands is None:
        target_bands = list(range(C))
    
    # 創建光譜掩碼
    spectral_mask = torch.zeros((B, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[:, band_idx, 0, 0] = 1.0
    
    # 參數化擾動
    delta = torch.zeros_like(images_unnorm, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分割任務，希望模型的預測與真實標籤不同
        return -F.nll_loss(F.log_softmax(outputs, dim=1), targets)

    for step in range(steps):
        optimizer.zero_grad()
        
        # 應用光譜掩碼到擾動
        masked_delta = delta * spectral_mask
        
        # 限制擾動在指定範圍內
        masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
        
        # 生成對抗樣本
        # adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
        adv_images = images_unnorm + masked_delta
        adv_images = images_unnorm + delta
        adv_images_norm = normalize(adv_images, mean, std)
        
        # 前向傳播
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 計算正則化損失（L2範數）
        reg_loss = c * (torch.sum(masked_delta ** 2) / B)
        
        # 總損失
        loss = adv_loss + reg_loss
        
        # 反向傳播
        loss.backward()
        optimizer.step()
    
    # 最終的掩碼擾動
    masked_delta = delta * spectral_mask
    masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
    
    # 生成最終對抗樣本
    # adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
    adv_images_unnorm = images_unnorm + masked_delta
    adv_images_norm = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images_norm = perceptual_constraint(
            images, adv_images_norm, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images_norm.detach()


def deepfool_spectral_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02, 
                           device='cuda', mean=None, std=None, target_bands=None,
                           apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                           use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                           use_spectral_importance=False, spectral_threshold=0.7,
                           **kwargs):
    """
    光譜域上的DeepFool攻擊
    
    在光譜域上進行攻擊時，專注於修改特定波段的光譜特徵，保持空間結構相對不變。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 類別數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過衝參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        target_bands (list): 要攻擊的目標波段，如果為None，會自動選擇重要波段
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
    # 計算自適應擾動大小
    if use_adaptive_eps:
        # 這裡假設我們傳入一個假的labels，實際上DeepFool不使用標籤
        dummy_labels = torch.zeros(images.size(0), images.size(2), images.size(3)).long().to(device)
        max_eps_tensor = adaptive_perturbation_size(images, dummy_labels, model, overshoot, min_eps, max_eps)
        max_eps_value = max_eps_tensor.view(-1)[0].item()  # 使用第一個樣本的擾動大小
    else:
        max_eps_value = None
    
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    B, C, H, W = images_unnorm.shape
    
    # 如果使用光譜重要性分析，優先使用分析結果
    if use_spectral_importance:
        band_importance = spectral_importance_analysis(images, labels, model, device)
        target_bands = []
        
        # 選擇重要性最高的波段直到達到閾值
        sorted_bands = torch.argsort(band_importance, descending=True)
        cumulative_importance = 0
        for band_idx in sorted_bands:
            target_bands.append(band_idx.item())
            cumulative_importance += band_importance[band_idx]
            if cumulative_importance >= spectral_threshold:
                break
    # 如果未指定目標波段，則自動選擇重要波段
    elif target_bands is None:
        target_bands = get_important_bands(images_unnorm, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))
    
    # 創建光譜掩碼
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0
    
    # 擴展掩碼到全尺寸
    spectral_mask = spectral_mask.repeat(B, 1, H, W)
    
    # 初始化對抗樣本
    adv_images_unnorm = images_unnorm.clone().detach()
    
    # 逐個樣本處理
    for i in range(B):
        # 獲取當前樣本
        image = images_unnorm[i:i+1].clone().detach().requires_grad_(True)
        
        # 獲取當前掩碼
        mask = spectral_mask[i:i+1]
        
        # 前向傳播
        outputs = model(normalize(image, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 獲取初始預測
        orig_class = outputs.argmax(1).item()
        
        # 初始化變量
        current_class = orig_class
        iteration = 0
        total_perturbation = torch.zeros_like(image)
        
        # DeepFool迭代
        while current_class == orig_class and iteration < max_iter:
            # 重置梯度
            if image.grad is not None:
                image.grad.zero_()
            
            # 前向傳播
            outputs = model(normalize(image, mean, std))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 獲取當前預測
            current_class = outputs.argmax(1).item()
            
            if current_class != orig_class:
                break
            
            # 計算每個類別的梯度和分數差異
            min_dist = float('inf')
            closest_class = None
            w_closest = None
            f_closest = None
            
            for k in range(num_classes):
                if k == orig_class:
                    continue
                
                # 計算類別k的得分和梯度
                outputs[0, k].backward(retain_graph=True)
                grad_k = image.grad.data.clone()
                
                # 應用光譜掩碼
                masked_grad_k = grad_k * mask
                
                # 重置梯度
                image.grad.zero_()
                
                # 計算原始類別的得分和梯度
                outputs[0, orig_class].backward(retain_graph=True)
                grad_orig = image.grad.data.clone()
                
                # 應用光譜掩碼
                masked_grad_orig = grad_orig * mask
                
                # 計算差異
                w = masked_grad_k - masked_grad_orig
                f = outputs[0, k].item() - outputs[0, orig_class].item()
                
                # 計算到決策邊界的距離
                dist = abs(f) / (torch.norm(w.view(-1)) + 1e-10)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
                    w_closest = w
                    f_closest = f
            
            # 重置梯度
            image.grad.zero_()
            
            if closest_class is None:
                break
            
            # 計算最小擾動
            perturbation = abs(f_closest) / (torch.norm(w_closest.view(-1)) + 1e-10)**2 * w_closest
            
            # 添加擾動
            perturbation = perturbation * (1 + overshoot)
            total_perturbation += perturbation
            
            # 限制擾動大小
            if use_adaptive_eps and max_eps_value is not None:
                perturbation_norm = torch.norm(total_perturbation)
                if perturbation_norm > max_eps_value:
                    total_perturbation = total_perturbation * max_eps_value / perturbation_norm
            
            # 更新圖像
            image = torch.clamp(images_unnorm[i:i+1] + total_perturbation, 0, 1).detach().requires_grad_(True)
            
            # 更新迭代計數
            iteration += 1
        
        # 更新批次中的對抗樣本
        adv_images_unnorm[i:i+1] = image.detach()
    
    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images = perceptual_constraint(
            images, adv_images, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images.detach() 