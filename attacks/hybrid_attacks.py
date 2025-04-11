import torch
import torch.nn.functional as F
from .utils import zero_gradients, get_important_bands
from .attack_utils import normalize, unnormalize
from .attack_utils import perceptual_constraint, adaptive_perturbation_size
from .attack_utils import spectral_importance_analysis, selective_perturbation

def fgsm_hybrid_attack(model, images, labels, eps, criterion, device, mean, std, 
                     spatial_weight=0.5, target_bands=None,
                     apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                     use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                     use_spectral_importance=False, spectral_threshold=0.7,
                     **kwargs):
    """
    結合空間域和光譜域的混合快速梯度符號法(FGSM)攻擊
    
    同時對空間結構和重要光譜波段進行攻擊，實現更強大且不易察覺的擾動。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像，形狀為 [B, C, H, W]
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        spatial_weight (float): 空間域攻擊權重，範圍[0,1]，預設0.5表示均等混合
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
    
    # 計算自適應擾動大小（如果需要）
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device).requires_grad_(True)

    B, C, H, W = images_norm.shape
    
    # Forward pass
    outputs = model(images_norm)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()

    # 取得梯度
    grad = images_norm.grad.data
    
    # 步驟1: 計算空間域部分
    # 計算空間梯度幅度 (在通道維度上取平均)
    spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
    
    # 創建空間掩碼：選擇梯度最大的空間位置
    spatial_mask = (spatial_grad_magnitude > torch.median(spatial_grad_magnitude)).float()
    
    # 步驟2: 計算光譜域部分
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
        target_bands = get_important_bands(images, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))

    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0

    # 步驟3: 混合兩種攻擊
    # 空間掩碼擴展到所有通道
    expanded_spatial_mask = spatial_mask.repeat(1, C, 1, 1)
    
    # 混合掩碼 = 空間權重 * 空間掩碼 + (1-空間權重) * 光譜掩碼
    # 注意這裡使用element-wise最大值而不是簡單加權，以保證攻擊強度
    hybrid_mask = torch.max(
        spatial_weight * expanded_spatial_mask,
        (1 - spatial_weight) * spectral_mask
    )
    
    # 計算符號梯度並應用混合掩碼
    grad_sign = torch.sign(grad)
    masked_grad_sign = grad_sign * hybrid_mask
    
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
    print(f"混合FGSM: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"混合FGSM: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"混合FGSM: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    print(f"混合FGSM: 空間權重: {spatial_weight}, 選擇的波段數量: {len(target_bands)}/{C}")
    
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
        print(f"混合FGSM: 攻擊成功率: {success_rate.item():.4f}")
    
    return adv_images_norm.detach()


def pgd_hybrid_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std,
                     spatial_weight=0.5, target_bands=None,
                     apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                     use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                     use_spectral_importance=False, spectral_threshold=0.7,
                     **kwargs):
    """
    結合空間域和光譜域的混合投影梯度下降(PGD)攻擊
    
    同時對空間結構和重要光譜波段進行迭代攻擊，實現更強大且不易察覺的擾動。
    
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
        spatial_weight (float): 空間域攻擊權重，範圍[0,1]，預設0.5表示均等混合
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
    # 如果未指定目標波段，則自動選擇重要波段
    elif target_bands is None:
        target_bands = get_important_bands(images_orig, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))

    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0
    
    # 保存空間擾動掩碼 (在PGD的首次迭代中計算)
    spatial_mask = None

    for i in range(steps):
        adv_images_unnorm.requires_grad = True
        outputs = model(normalize(adv_images_unnorm, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # 計算空間梯度幅度
        grad = adv_images_unnorm.grad.data
        
        # 首次迭代：計算空間掩碼
        if spatial_mask is None:
            spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
            
            # 選擇梯度最大的前25%像素位置
            threshold = torch.quantile(spatial_grad_magnitude.view(B, -1), 0.75, dim=1).view(B, 1, 1, 1)
            spatial_mask = (spatial_grad_magnitude > threshold).float().repeat(1, C, 1, 1)
        
        # 混合掩碼 = 空間權重 * 空間掩碼 + (1-空間權重) * 光譜掩碼
        # 使用element-wise最大值操作來合併掩碼
        hybrid_mask = torch.max(
            spatial_weight * spatial_mask,
            (1 - spatial_weight) * spectral_mask.repeat(B, 1, 1, 1)
        )
        
        # 應用混合掩碼到梯度
        masked_grad = grad * hybrid_mask
        grad_sign = masked_grad.sign()
        
        # 更新對抗樣本
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        
        # 確保擾動在限制範圍內
        if use_adaptive_eps:
            # 根據每個樣本的自適應擾動大小計算限制
            eta = torch.clamp(adv_images_unnorm - images_orig, 
                           min=-adaptive_eps.squeeze(-1).squeeze(-1), 
                           max=adaptive_eps.squeeze(-1).squeeze(-1))
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


def cw_hybrid_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1,
                    device='cuda', mean=None, std=None, spatial_weight=0.5, target_bands=None,
                    apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                    use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                    use_spectral_importance=False, spectral_threshold=0.7,
                    **kwargs):
    """
    結合空間域和光譜域的混合Carlini & Wagner攻擊
    
    同時優化空間結構擾動和光譜擾動，實現高效且不易察覺的對抗樣本。
    
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
        spatial_weight (float): 空間域攻擊權重，範圍[0,1]，預設0.5表示均等混合
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
    # 如果未指定目標波段，則自動選擇重要波段
    elif target_bands is None:
        target_bands = get_important_bands(images_unnorm, n_bands=C//3)
    
    # 創建兩種參數化擾動：
    # 1. 空間擾動：每個空間位置共享同一個擾動值
    spatial_delta = torch.zeros((B, 1, H, W), requires_grad=True, device=device)
    
    # 2. 光譜擾動：只在指定波段上有擾動
    spectral_delta = torch.zeros_like(images_unnorm, requires_grad=True, device=device)
    
    # 光譜掩碼：標記要攻擊的波段
    spectral_mask = torch.zeros((B, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[:, band_idx, 0, 0] = 1.0
    
    # 優化兩種擾動
    optimizer = torch.optim.Adam([spatial_delta, spectral_delta], lr=lr)
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分割任務，希望模型的預測與真實標籤不同
        return -F.nll_loss(F.log_softmax(outputs, dim=1), targets)

    for step in range(steps):
        optimizer.zero_grad()
        
        # 1. 計算空間擾動部分（擴展到所有通道）
        expanded_spatial_delta = spatial_delta.repeat(1, C, 1, 1)
        
        # 2. 計算光譜擾動部分（只擾動選定的波段）
        masked_spectral_delta = spectral_delta * spectral_mask
        
        # 3. 混合兩種擾動，使用加權合併
        hybrid_delta = spatial_weight * expanded_spatial_delta + (1 - spatial_weight) * masked_spectral_delta
        
        # 限制擾動在指定範圍內
        hybrid_delta = torch.clamp(hybrid_delta, -eps_to_use, eps_to_use)
        
        # 生成對抗樣本
        # adv_images = torch.clamp(images_unnorm + hybrid_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
        adv_images_unnorm = images_unnorm + hybrid_delta
        adv_images_norm = normalize(adv_images_unnorm, mean, std)
        
        # 前向傳播
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 計算正則化損失（L2範數）
        reg_loss = c * (torch.sum(hybrid_delta ** 2) / B)
        
        # 總損失
        loss = adv_loss + reg_loss
        
        # 反向傳播
        loss.backward()
        optimizer.step()
    
    # 計算最終的混合擾動
    expanded_spatial_delta = spatial_delta.repeat(1, C, 1, 1)
    masked_spectral_delta = spectral_delta * spectral_mask
    hybrid_delta = spatial_weight * expanded_spatial_delta + (1 - spatial_weight) * masked_spectral_delta
    
    # 限制擾動在指定範圍內
    hybrid_delta = torch.clamp(hybrid_delta, -eps_to_use, eps_to_use)
    
    # 生成最終對抗樣本
    # adv_images = torch.clamp(images_unnorm + hybrid_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
    adv_images_unnorm = images_unnorm + hybrid_delta
    adv_images_norm = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images_norm = perceptual_constraint(
            images, adv_images_norm, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images_norm.detach()


def deepfool_hybrid_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02,
                           device='cuda', mean=None, std=None, spatial_weight=0.5, target_bands=None,
                           apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                           use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                           use_spectral_importance=False, spectral_threshold=0.7,
                           **kwargs):
    """
    結合空間域和光譜域的混合DeepFool攻擊
    
    通過迭代尋找最小擾動，同時考慮空間和光譜特性，生成高質量對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 類別數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過衝參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        spatial_weight (float): 空間域攻擊權重，範圍[0,1]，預設0.5表示均等混合
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
    
    # 初始化對抗樣本
    adv_images_unnorm = images_unnorm.clone().detach()
    
    # 計算空間掩碼
    spatial_masks = []
    for i in range(B):
        sample = images_unnorm[i:i+1].clone().detach().requires_grad_(True)
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 使用最大類別的梯度
        pred = outputs.argmax(1)
        loss = F.cross_entropy(outputs, pred)
        
        model.zero_grad()
        loss.backward()
        
        grad = sample.grad.data
        spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
        
        # 選擇梯度最大的前25%像素位置
        threshold = torch.quantile(spatial_grad_magnitude.view(1, -1), 0.75, dim=1).view(1, 1, 1, 1)
        spatial_mask_i = (spatial_grad_magnitude > threshold).float()
        spatial_masks.append(spatial_mask_i)
    
    # 將所有空間掩碼拼接為批次
    spatial_mask_batch = torch.cat(spatial_masks, dim=0)
    
    # 對每個樣本逐一進行DeepFool迭代
    for i in range(B):
        sample = adv_images_unnorm[i:i+1].clone().detach().requires_grad_(True)
        sample_original = images_unnorm[i:i+1].clone().detach()
        
        # 獲取當前樣本的空間掩碼
        s_mask = spatial_mask_batch[i:i+1].repeat(1, C, 1, 1)
        
        # 獲取擴展的光譜掩碼
        sp_mask = spectral_mask.repeat(1, 1, H, W)
        
        # 混合掩碼
        hybrid_mask = torch.max(
            spatial_weight * s_mask,
            (1 - spatial_weight) * sp_mask
        )
        
        # 初始化變量
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        original_pred = outputs.argmax(1).item()
        
        current_pred = original_pred
        iteration = 0
        total_perturbation = torch.zeros_like(sample)
        
        # DeepFool迭代
        while current_pred == original_pred and iteration < max_iter:
            # 重置梯度
            if sample.grad is not None:
                sample.grad.zero_()
            
            # 計算梯度
            outputs = model(normalize(sample, mean, std))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 如果預測已經改變，退出循環
            current_pred = outputs.argmax(1).item()
            if current_pred != original_pred:
                break
            
            # 找到最近的決策邊界
            min_dist = float('inf')
            closest_class = None
            w_closest = None
            f_closest = None
            
            # 檢查所有其他類別
            for k in range(num_classes):
                if k == original_pred:
                    continue
                
                zero_gradients(sample)
                outputs[0, k].backward(retain_graph=True)
                grad_k = sample.grad.data.clone()
                
                zero_gradients(sample)
                outputs[0, original_pred].backward(retain_graph=True)
                grad_orig = sample.grad.data.clone()
                
                # 計算梯度差並應用混合掩碼
                masked_grad_diff = (grad_k - grad_orig) * hybrid_mask
                
                # 計算輸出差異
                score_diff = outputs[0, k].item() - outputs[0, original_pred].item()
                
                # 計算到決策邊界的距離
                dist = abs(score_diff) / (torch.norm(masked_grad_diff.view(-1)) + 1e-10)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
                    w_closest = masked_grad_diff
                    f_closest = score_diff
            
            # 如果找不到最近的類別，退出循環
            if closest_class is None:
                break
            
            # 計算擾動
            perturbation = -f_closest * w_closest / (torch.norm(w_closest.view(-1))**2 + 1e-10)
            
            # 添加擾動（帶過衝）
            perturbation = perturbation * (1 + overshoot)
            total_perturbation = total_perturbation + perturbation
            
            # 如果使用自適應擾動大小，確保擾動不超過限制
            if use_adaptive_eps and max_eps_value is not None:
                perturbation_norm = torch.norm(total_perturbation)
                if perturbation_norm > max_eps_value:
                    total_perturbation = total_perturbation * max_eps_value / perturbation_norm
            
            # 更新樣本
            # sample = torch.clamp(sample_original + total_perturbation, 0, 1)
            sample = sample_original + total_perturbation.detach().requires_grad_(True)
            
            # 更新迭代計數
            iteration += 1
        
        # 更新批次
        adv_images_unnorm[i:i+1] = sample.detach()
    
    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images = perceptual_constraint(
            images, adv_images, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images.detach()
 
 