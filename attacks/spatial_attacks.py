import torch
import torch.nn.functional as F
from .utils import zero_gradients, get_important_bands
from .attack_utils import normalize, unnormalize
from .attack_utils import perceptual_constraint, adaptive_perturbation_size
from .attack_utils import spectral_importance_analysis, selective_perturbation

def fgsm_spatial_attack(model, images, labels, eps, criterion, device, mean, std,
                       apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                       use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                       use_spectral_importance=False, spectral_threshold=0.7,
                       **kwargs):
    """
    空間域快速梯度符號法(FGSM)攻擊
    
    專注於擾動圖像中的重要空間結構，實現高效且針對性的對抗樣本生成。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像，形狀為 [B, C, H, W]
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
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
    
    # 計算梯度重要性（針對空間位置而非通道）
    # 對所有通道的梯度取絕對值並求平均
    grad_importance = torch.mean(torch.abs(grad), dim=1, keepdim=True)
    
    # 增加包含的像素量：選取重要性最高的前50%（而不是30%）
    importance_threshold = torch.quantile(grad_importance.view(grad_importance.size(0), -1), 0.5, dim=1)
    importance_threshold = importance_threshold.view(grad_importance.size(0), 1, 1, 1)
    
    # 創建空間掩碼
    spatial_mask = (grad_importance >= importance_threshold).float()
    
    # 將掩碼擴展到所有通道
    expanded_mask = spatial_mask.repeat(1, grad.size(1), 1, 1)
    
    # 使用符號梯度和掩碼生成擾動
    grad_sign = torch.sign(grad)
    
    # 應用掩碼
    masked_grad_sign = grad_sign * expanded_mask
    
    # 使用自適應擾動大小或固定擾動大小
    if use_adaptive_eps:
        # 為每個樣本單獨應用擾動
        batch_size = images_norm.size(0)
        perturbation = torch.zeros_like(masked_grad_sign)
        for b in range(batch_size):
            perturbation[b] = adaptive_eps[b, 0, 0, 0].item() * masked_grad_sign[b]
    else:
        perturbation = effective_eps * masked_grad_sign
    
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
        actual_perturbation_l_inf = torch.norm(actual_perturbation.view(B, -1), p=float('inf'), dim=1)
        actual_perturbation_l2 = torch.norm(actual_perturbation.view(B, -1), p=2, dim=1)
    
    # 輸出調試信息
    perturbation_l_inf = torch.norm(perturbation.view(B, -1), p=float('inf'), dim=1)
    print(f"空間FGSM: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"空間FGSM: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"空間FGSM: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    print(f"空間FGSM: 選擇的像素比例: {spatial_mask.mean().item():.4f}")
    
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
        print(f"空間FGSM: 攻擊成功率: {success_rate.item():.4f}")
    
    return adv_images_norm.detach()


def pgd_spatial_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std,
                      apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                      use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                      use_spectral_importance=False, spectral_threshold=0.7,
                      **kwargs):
    """
    空間域投影梯度下降(PGD)攻擊
    
    通過多步優化，針對空間結構生成對抗樣本，實現更強效的誤導。
    
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
    
    # 初始化空間掩碼（首次迭代時計算）
    spatial_mask = None
    
    for i in range(steps):
        adv_images_unnorm.requires_grad = True
        outputs = model(normalize(adv_images_unnorm, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_images_unnorm.grad.data
        
        # 首次迭代時計算空間掩碼
        if spatial_mask is None:
            # 計算梯度重要性
            grad_importance = torch.mean(torch.abs(grad), dim=1, keepdim=True)
            
            # 選擇重要的空間位置
            importance_threshold = torch.quantile(grad_importance.view(grad_importance.size(0), -1), 0.7, dim=1)
            importance_threshold = importance_threshold.view(grad_importance.size(0), 1, 1, 1)
            spatial_mask = (grad_importance >= importance_threshold).float()
            
            # 擴展到所有通道
            spatial_mask = spatial_mask.repeat(1, grad.size(1), 1, 1)
        
        # 應用空間掩碼
        masked_grad = grad * spatial_mask
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


def cw_spatial_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1,
                    device='cuda', mean=None, std=None,
                    apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                    use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                    use_spectral_importance=False, spectral_threshold=0.7,
                    **kwargs):
    """
    空間域Carlini & Wagner攻擊
    
    通過優化L2範數最小的擾動，專注於空間位置，實現高質量的對抗樣本。
    
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
        # 安全地提取每個樣本的擾動大小
        eps_values = []
        for b in range(images.size(0)):
            eps_values.append(adaptive_eps_tensor[b, 0, 0, 0].item())
        # 使用平均值作為整體擾動大小
        eps_to_use = sum(eps_values) / len(eps_values) if eps_values else eps
    else:
        eps_to_use = eps
    
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    B, C, H, W = images_unnorm.shape
    
    # 計算空間重要性掩碼
    images_copy = images_unnorm.clone().detach().requires_grad_(True)
    outputs = model(normalize(images_copy, mean, std))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_importance = torch.mean(torch.abs(images_copy.grad.data), dim=1, keepdim=True)
    importance_threshold = torch.quantile(grad_importance.view(B, -1), 0.7, dim=1).view(B, 1, 1, 1)
    spatial_mask = (grad_importance >= importance_threshold).float().repeat(1, C, 1, 1)
    
    # 參數化擾動
    delta = torch.zeros_like(images_unnorm, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分割任務，希望模型的預測與真實標籤不同
        return -F.nll_loss(F.log_softmax(outputs, dim=1), targets)

    for step in range(steps):
        optimizer.zero_grad()
        
        # 應用空間掩碼到擾動
        masked_delta = delta * spatial_mask
        
        # 限制擾動在指定範圍內
        masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
        
        # 投影回有效範圍，只限制擾動而非最終圖像
        masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
        
        # 添加擾動到原始圖像
        adv_images_unnorm = images_unnorm + masked_delta
        
        # 移除對adv_images的clamp操作
        # adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
        adv_images = images_unnorm + masked_delta
        adv_images_norm = normalize(adv_images_unnorm, mean, std)
        
        # 前向傳播
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失和正則化損失
        adv_loss = compute_adv_loss(outputs, labels)
        reg_loss = c * (torch.sum(masked_delta ** 2) / B)
        
        # 總損失
        loss = adv_loss + reg_loss
        
        # 反向傳播
        loss.backward()
        optimizer.step()
    
    # 最終的掩碼擾動
    masked_delta = delta * spatial_mask
    masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
    
    # 投影回有效範圍，只限制擾動而非最終圖像
    masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
    
    # 添加擾動到原始圖像
    adv_images_unnorm = images_unnorm + masked_delta
    
    # 移除對adv_images的clamp操作
    # adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)  # 移除clamp操作，保持原始數據範圍
    adv_images = images_unnorm + masked_delta
    adv_images_norm = normalize(adv_images_unnorm, mean, std)
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images_norm = perceptual_constraint(
            images, adv_images_norm, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images_norm.detach()


def deepfool_spatial_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02,
                           device='cuda', mean=None, std=None,
                           apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                           use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                           use_spectral_importance=False, spectral_threshold=0.7,
                           **kwargs):
    """
    空間域DeepFool攻擊
    
    通過迭代尋找最小擾動，優先考慮空間重要性，生成高質量對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 類別數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過衝參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
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
        # 安全地提取擾動大小
        max_eps_values = []
        for b in range(images.size(0)):
            max_eps_values.append(max_eps_tensor[b, 0, 0, 0].item())
        # 使用平均值作為整體擾動大小
        max_eps_value = sum(max_eps_values) / len(max_eps_values) if max_eps_values else None
    else:
        max_eps_value = None
    
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    B, C, H, W = images_unnorm.shape
    
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
        
        # 選擇梯度最大的前30%像素位置
        threshold = torch.quantile(spatial_grad_magnitude.view(1, -1), 0.7, dim=1).view(1, 1, 1, 1)
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
                
                # 計算梯度差並應用空間掩碼
                masked_grad_diff = (grad_k - grad_orig) * s_mask
                
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