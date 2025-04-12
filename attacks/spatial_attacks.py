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
    effective_alpha = alpha * data_range_factor
    
    # 如果使用自適應擾動大小，也需要放大
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device)
    adv_images_norm = images_norm.clone().detach()
    
    B, C, H, W = images_norm.shape
    
    # 初始化空間掩碼（首次迭代時計算）
    spatial_mask = None
    
    for i in range(steps):
        adv_images_norm.requires_grad = True
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        grad = adv_images_norm.grad.data
        
        # 首次迭代時計算空間掩碼
        if spatial_mask is None:
            # 計算梯度重要性
            grad_importance = torch.mean(torch.abs(grad), dim=1, keepdim=True)
            
            # 選擇重要的空間位置，增加到50%
            importance_threshold = torch.quantile(grad_importance.view(grad_importance.size(0), -1), 0.5, dim=1)
            importance_threshold = importance_threshold.view(grad_importance.size(0), 1, 1, 1)
            spatial_mask = (grad_importance >= importance_threshold).float()
            
            # 擴展到所有通道
            spatial_mask = spatial_mask.repeat(1, grad.size(1), 1, 1)
        
        # 應用空間掩碼
        masked_grad = grad * spatial_mask
        grad_sign = masked_grad.sign()
        
        # 使用自適應擾動大小或固定擾動大小
        if use_adaptive_eps:
            # 為每個樣本單獨應用擾動
            step_perturbation = torch.zeros_like(grad_sign)
            for b in range(B):
                step_size = (adaptive_eps[b, 0, 0, 0].item() / steps) * 2  # 稍微更激進的步長
                step_perturbation[b] = step_size * grad_sign[b]
        else:
            step_perturbation = effective_alpha * grad_sign
        
        # 更新對抗樣本
        adv_images_norm = adv_images_norm.detach() + step_perturbation
        
        # 確保擾動在限制範圍內
        if use_adaptive_eps:
            # 每個樣本分別限制擾動
            for b in range(B):
                eta = adv_images_norm[b] - images_norm[b]
                eta = torch.clamp(eta, -adaptive_eps[b, 0, 0, 0].item(), adaptive_eps[b, 0, 0, 0].item())
                adv_images_norm[b] = images_norm[b] + eta
        else:
            eta = adv_images_norm - images_norm
            eta = torch.clamp(eta, -effective_eps, effective_eps)
            adv_images_norm = images_norm + eta
    
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
    perturbation_l_inf = torch.norm((adv_images_norm - images_norm).view(B, -1), p=float('inf'), dim=1)
    print(f"空間PGD: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"空間PGD: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"空間PGD: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    print(f"空間PGD: 選擇的像素比例: {spatial_mask.mean().item():.4f}")
    
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
        print(f"空間PGD: 攻擊成功率: {success_rate.item():.4f}")
    
    return adv_images_norm.detach()


def cw_spatial_attack(model, images, labels, c=0.001, kappa=10.0, steps=2000, lr=0.01, eps=0.1,
                    device='cuda', mean=None, std=None,
                    apply_perceptual_constraint=False, ssim_threshold=0.95, lpips_threshold=0.05,
                    use_adaptive_eps=False, min_eps=0.01, max_eps=0.1,
                    use_spectral_importance=False, spectral_threshold=0.5,
                    **kwargs):
    """
    空間域Carlini & Wagner攻擊
    
    通過優化L2範數最小的擾動，專注於空間位置，實現高質量的對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        c (float): 平衡係數，預設0.001
        kappa (float): 置信度參數，預設10.0，較大的值表示更強的分類錯誤要求
        steps (int): 優化步數，預設2000
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
        spectral_threshold (float): 光譜重要性閾值，預設0.5
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 添加數據範圍自適應的擾動放大因子
    # 高光譜數據通常範圍很大，需要更大的擾動才能產生明顯效果
    data_range_factor = 20.0  # 基本放大因子，提高最小值
    
    # 根據數據估算合適的放大因子
    with torch.no_grad():
        orig_images_unnorm = unnormalize(images.clone(), mean, std)
        data_range = torch.max(orig_images_unnorm) - torch.min(orig_images_unnorm)
        if data_range > 1.0:  # 高光譜數據通常範圍較大
            # 根據數據範圍動態調整放大因子，最小20倍，最大100倍
            data_range_factor = max(min(data_range / 5.0, 100.0), 20.0)
    
    print(f"空間CW: 使用數據範圍自適應放大因子: {data_range_factor:.2f}, 數據範圍: {data_range:.2f}")
    
    # 應用數據範圍因子到擾動上和學習率上
    effective_eps = eps * data_range_factor
    effective_lr = lr * data_range_factor  # 擴大學習率以加速收斂
    effective_c = c / (data_range_factor * 2)  # 進一步減小正則化係數
    
    # 如果使用自適應擾動大小，也需要放大
    if use_adaptive_eps:
        min_eps = min_eps * data_range_factor
        max_eps = max_eps * data_range_factor
        adaptive_eps = adaptive_perturbation_size(images, labels, model, effective_eps, min_eps, max_eps)
        # 安全地提取每個樣本的擾動大小
        eps_values = []
        for b in range(images.size(0)):
            eps_values.append(adaptive_eps[b, 0, 0, 0].item())
        # 使用平均值作為整體擾動大小
        eps_to_use = sum(eps_values) / len(eps_values) if eps_values else effective_eps
    else:
        eps_to_use = effective_eps
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device)
    
    B, C, H, W = images_norm.shape
    
    # 計算空間重要性掩碼（在正規化域中）
    images_copy = images_norm.clone().detach().requires_grad_(True)
    outputs = model(images_copy)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_importance = torch.mean(torch.abs(images_copy.grad.data), dim=1, keepdim=True)
    # 選擇重要的空間位置，從0.5改為0.3，更集中地選擇重要區域
    importance_threshold = torch.quantile(grad_importance.view(B, -1), 0.3, dim=1).view(B, 1, 1, 1)
    spatial_mask = (grad_importance >= importance_threshold).float().repeat(1, C, 1, 1)
    print(f"空間CW: 選擇的像素比例: {spatial_mask.mean().item():.4f}")
    
    # 參數化擾動（在正規化域中）
    delta = torch.zeros_like(images_norm, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=effective_lr)
    
    # 用於計算對抗損失的函數 - 增強版
    def compute_adv_loss(outputs, targets):
        # 強調類別變化的重要性
        # 處理分割模型輸出 [B, C, H, W]
        B, C, H, W = outputs.shape
        
        # 為分割模型重新設計損失函數
        # 首先將輸出從[B, C, H, W]轉換為[B*H*W, C]
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
        
        # 將標籤從[B, H, W]轉換為[B*H*W]
        targets_flat = targets.reshape(-1)
        
        # 排除標籤中的無效值（如果有）
        valid_mask = (targets_flat >= 0) & (targets_flat < C)
        if valid_mask.sum() > 0:
            outputs_flat = outputs_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]
        
        # 計算每個像素的softmax概率
        probs = F.softmax(outputs_flat, dim=1)
        
        # 獲取目標類別的概率和最大非目標類別的概率
        target_probs = torch.gather(probs, 1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # 創建one-hot編碼，並反轉它以獲取非目標類別的掩碼
        non_target_mask = 1 - F.one_hot(targets_flat, probs.size(1)).float()
        
        # 使用掩碼獲取非目標類別的概率，並找出最大值
        # 添加小的epsilon防止數值問題
        max_other_probs, _ = torch.max(probs * non_target_mask + 1e-7 * (1 - non_target_mask), dim=1)
        
        # 目標是增加最大非目標類別的概率，減少目標類別的概率
        margin_loss = torch.clamp(target_probs - max_other_probs + kappa, min=0)
        
        # 返回負的平均邊界損失，我們希望最小化目標類別概率
        return -margin_loss.mean()

    # 跟踪最佳對抗樣本
    best_loss = float('inf')
    best_delta = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # 應用空間掩碼到擾動
        masked_delta = delta * spatial_mask
        
        # 限制擾動在指定範圍內
        masked_delta = torch.clamp(masked_delta, -eps_to_use, eps_to_use)
        
        # 添加擾動到原始圖像
        adv_images_norm = images_norm + masked_delta
        
        # 前向傳播
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失和正則化損失
        adv_loss = compute_adv_loss(outputs, labels)
        reg_loss = effective_c * (torch.sum(masked_delta ** 2) / B)
        
        # 總損失
        total_loss = adv_loss + reg_loss
        
        # 保存最佳結果
        if adv_loss.item() < best_loss:
            best_loss = adv_loss.item()
            best_delta = masked_delta.clone().detach()
            
            # 每100步或最後一步輸出當前進度
            if step % 100 == 0 or step == steps - 1:
                success_rate = evaluate_adv_success(model, images_norm, adv_images_norm)
                print(f"空間CW: 步驟 {step}/{steps}, 對抗損失: {adv_loss.item():.4f}, 正則化損失: {reg_loss.item():.4f}, 成功率: {success_rate:.4f}")
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
    
    # 使用最佳擾動
    if best_delta is not None:
        final_delta = best_delta
    else:
        # 如果沒有找到更好的擾動，使用最後一個
        final_delta = masked_delta.detach()
    
    # 創建最終對抗樣本
    adv_images_norm = torch.clamp(images_norm + final_delta, 0, 1)
    
    # 計算原始域的擾動大小
    with torch.no_grad():
        images_unnorm = unnormalize(images, mean, std)
        adv_images_unnorm = unnormalize(adv_images_norm, mean, std)
        actual_perturbation = adv_images_unnorm - images_unnorm
        actual_perturbation_l_inf = torch.norm(actual_perturbation.view(B, -1), p=float('inf'), dim=1)
        actual_perturbation_l2 = torch.norm(actual_perturbation.view(B, -1), p=2, dim=1)
    
    # 輸出調試信息
    perturbation_l_inf = torch.norm(final_delta.view(B, -1), p=float('inf'), dim=1)
    print(f"空間CW: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"空間CW: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"空間CW: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    
    # 測試攻擊效果
    with torch.no_grad():
        success_rate = evaluate_adv_success(model, images_norm, adv_images_norm)
        print(f"空間CW: 最終攻擊成功率: {success_rate:.4f}")
    
    # 應用感知限制
    if apply_perceptual_constraint:
        adv_images_norm = perceptual_constraint(
            images, adv_images_norm, ssim_threshold, lpips_threshold, device
        )
    
    return adv_images_norm.detach()

# 輔助函數：評估對抗攻擊成功率
def evaluate_adv_success(model, orig_images, adv_images):
    with torch.no_grad():
        orig_output = model(orig_images)
        adv_output = model(adv_images)
        if isinstance(orig_output, tuple):
            orig_output = orig_output[0]
        if isinstance(adv_output, tuple):
            adv_output = adv_output[0]
        orig_pred = orig_output.argmax(1)
        adv_pred = adv_output.argmax(1)
        success_rate = (orig_pred != adv_pred).float().mean().item()
    return success_rate


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
    # 添加數據範圍自適應的擾動放大因子
    # 高光譜數據通常範圍很大，需要更大的擾動才能產生明顯效果
    data_range_factor = 20.0  # 基本放大因子，提高最小值
    
    # 根據數據估算合適的放大因子
    with torch.no_grad():
        orig_images_unnorm = unnormalize(images.clone(), mean, std)
        data_range = torch.max(orig_images_unnorm) - torch.min(orig_images_unnorm)
        if data_range > 1.0:  # 高光譜數據通常範圍較大
            # 根據數據範圍動態調整放大因子，最小20倍，最大100倍
            data_range_factor = max(min(data_range / 5.0, 100.0), 20.0)
    
    print(f"空間DeepFool: 使用數據範圍自適應放大因子: {data_range_factor:.2f}, 數據範圍: {data_range:.2f}")
    
    # 計算自適應擾動大小
    if use_adaptive_eps:
        # 這裡假設我們傳入一個假的labels，實際上DeepFool不使用標籤
        dummy_labels = torch.zeros(images.size(0), images.size(2), images.size(3)).long().to(device)
        
        # 應用放大因子到擾動參數
        effective_overshoot = overshoot * data_range_factor
        effective_min_eps = min_eps * data_range_factor
        effective_max_eps = max_eps * data_range_factor
        
        max_eps_tensor = adaptive_perturbation_size(images, dummy_labels, model, effective_overshoot, effective_min_eps, effective_max_eps)
        # 安全地提取擾動大小
        max_eps_values = []
        for b in range(images.size(0)):
            max_eps_values.append(max_eps_tensor[b, 0, 0, 0].item())
        # 使用平均值作為整體擾動大小
        max_eps_value = sum(max_eps_values) / len(max_eps_values) if max_eps_values else None
    else:
        # 即使不使用自適應擾動大小，也應用放大因子到過衝參數
        effective_overshoot = overshoot * data_range_factor
        max_eps_value = None
    
    # 直接在正規化域中操作
    images_norm = images.clone().detach().to(device)
    B, C, H, W = images_norm.shape
    
    # 初始化對抗樣本
    adv_images_norm = images_norm.clone().detach()
    
    # 計算空間掩碼
    spatial_masks = []
    for i in range(B):
        sample = images_norm[i:i+1].clone().detach().requires_grad_(True)
        outputs = model(sample)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 處理分割模型輸出
        if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
            # 分割模型輸出 - 計算每個類別的平均置信度
            class_confidence = outputs.mean(dim=(2, 3))  # [B, C]
            pred = class_confidence.argmax(1)
        else:
            # 分類模型輸出
            pred = outputs.argmax(1)
        
        # 計算適合模型輸出的損失
        if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
            # 分割模型 - 計算空間平均交叉熵
            outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))
            pred_flat = pred.repeat(H, W).view(-1)
            loss = F.cross_entropy(outputs_flat, pred_flat)
        else:
            # 分類模型 - 標準交叉熵
            loss = F.cross_entropy(outputs, pred)
        
        model.zero_grad()
        loss.backward()
        
        grad = sample.grad.data
        spatial_grad_magnitude = torch.mean(torch.abs(grad), dim=1, keepdim=True)
        
        # 選擇梯度最大的前30%像素位置
        threshold = torch.quantile(spatial_grad_magnitude.view(1, -1), 0.7, dim=1).view(1, 1, 1, 1)
        spatial_mask_i = (spatial_grad_magnitude > threshold).float()
        spatial_masks.append(spatial_mask_i)
    
    # 將所有空間掩碼拼接為批次
    spatial_mask_batch = torch.cat(spatial_masks, dim=0)
    
    # 對每個樣本逐一進行DeepFool迭代
    for i in range(B):
        sample = adv_images_norm[i:i+1].clone().detach().requires_grad_(True)
        sample_original = images_norm[i:i+1].clone().detach()
        
        # 獲取當前樣本的空間掩碼
        s_mask = spatial_mask_batch[i:i+1].repeat(1, C, 1, 1)
        
        # 初始化變量
        outputs = model(sample)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 處理分割模型輸出
        if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
            # 分割模型輸出 - 計算每個類別的平均置信度
            class_confidence = outputs.mean(dim=(2, 3))  # [B, C]
            original_pred = class_confidence.argmax(1).item()
        else:
            # 分類模型輸出
            original_pred = outputs.argmax(1).item()
        
        current_pred = original_pred
        iteration = 0
        total_perturbation = torch.zeros_like(sample)
        
        # DeepFool迭代
        while current_pred == original_pred and iteration < max_iter:
            # 創建一個新的計算圖
            sample_iter = sample.clone().detach().requires_grad_(True)
            
            # 計算梯度
            outputs = model(sample_iter)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 處理分割模型輸出
            if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
                # 分割模型輸出 - 使用空間平均
                class_confidence = outputs.mean(dim=(2, 3))  # [B, C]
                current_pred = class_confidence.argmax(1).item()
            else:
                # 分類模型輸出
                current_pred = outputs.argmax(1).item()
            
            # 如果預測已經改變，退出循環
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
                
                # 對k類別計算梯度
                if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
                    # 分割模型
                    class_confidence = outputs.mean(dim=(2, 3))
                    # 創建新的計算圖以正確計算梯度
                    out_k = class_confidence[0, k]
                else:
                    # 分類模型
                    out_k = outputs[0, k]
                
                # 清除之前的梯度
                model.zero_grad()
                if sample_iter.grad is not None:
                    sample_iter.grad.zero_()
                
                # 計算k類別的梯度
                out_k.backward(retain_graph=True)
                if sample_iter.grad is None:
                    print("警告: k類別的梯度為None")
                    continue
                    
                grad_k = sample_iter.grad.clone()
                
                # 對原始預測類別計算梯度
                if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
                    # 分割模型
                    class_confidence = outputs.mean(dim=(2, 3))
                    out_orig = class_confidence[0, original_pred]
                else:
                    # 分類模型
                    out_orig = outputs[0, original_pred]
                
                # 清除之前的梯度
                model.zero_grad()
                if sample_iter.grad is not None:
                    sample_iter.grad.zero_()
                
                # 計算original_pred類別的梯度
                out_orig.backward(retain_graph=True)
                if sample_iter.grad is None:
                    print("警告: 原始類別的梯度為None")
                    continue
                    
                grad_orig = sample_iter.grad.clone()
                
                # 計算梯度差並應用空間掩碼
                masked_grad_diff = (grad_k - grad_orig) * s_mask
                
                # 計算輸出差異
                if outputs.dim() == 4 and outputs.size(1) > 1 and outputs.size(2) == H and outputs.size(3) == W:
                    # 分割模型
                    class_confidence = outputs.mean(dim=(2, 3))
                    score_diff = class_confidence[0, k].item() - class_confidence[0, original_pred].item()
                else:
                    # 分類模型
                    score_diff = outputs[0, k].item() - outputs[0, original_pred].item()
                
                # 計算到決策邊界的距離
                norm_value = torch.norm(masked_grad_diff.view(-1))
                if norm_value < 1e-10:
                    continue  # 避免除以接近零的值
                
                dist = abs(score_diff) / norm_value
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
                    w_closest = masked_grad_diff
                    f_closest = score_diff
            
            # 如果找不到最近的類別，退出循環
            if closest_class is None:
                break
            
            # 計算擾動
            norm_squared = torch.norm(w_closest.view(-1))**2
            if norm_squared < 1e-10:
                break  # 避免除以接近零的值
            
            perturbation = -f_closest * w_closest / norm_squared
            
            # 添加擾動（帶過衝）
            perturbation = perturbation * (1 + effective_overshoot)
            total_perturbation = total_perturbation + perturbation
            
            # 如果使用自適應擾動大小，確保擾動不超過限制
            if use_adaptive_eps and max_eps_value is not None:
                perturbation_norm = torch.norm(total_perturbation)
                if perturbation_norm > max_eps_value:
                    total_perturbation = total_perturbation * max_eps_value / perturbation_norm
            
            # 更新樣本
            sample = sample_original + total_perturbation.detach()
            
            # 更新迭代計數
            iteration += 1
        
        # 更新批次
        adv_images_norm[i:i+1] = sample.detach()
    
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
    perturbation_l_inf = torch.norm((adv_images_norm - images_norm).view(B, -1), p=float('inf'), dim=1)
    print(f"空間DeepFool: 正規化域平均L∞擾動大小: {perturbation_l_inf.mean().item():.4f}")
    print(f"空間DeepFool: 原始域平均L∞擾動大小: {actual_perturbation_l_inf.mean().item():.4f}")
    print(f"空間DeepFool: 原始域平均L2擾動大小: {actual_perturbation_l2.mean().item():.4f}")
    print(f"空間DeepFool: 選擇的像素比例: {spatial_mask_batch.mean().item():.4f}")
    
    # 測試攻擊效果
    with torch.no_grad():
        success_rate = evaluate_adv_success(model, images_norm, adv_images_norm)
        print(f"空間DeepFool: 攻擊成功率: {success_rate:.4f}")
    
    return adv_images_norm.detach() 