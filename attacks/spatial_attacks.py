import torch
import torch.nn.functional as F
from .utils import normalize, unnormalize, zero_gradients, get_important_bands

def fgsm_spatial_attack(model, images, labels, eps, criterion, device, mean, std):
    """
    快速梯度符號法(FGSM)空間攻擊
    
    專注於攻擊重要的空間結構，對圖像的空間特徵進行擾動。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像，形狀為 [B, C, H, W]
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 反正規化
    images_unnorm = unnormalize(images, mean, std)
    images_unnorm = images_unnorm.clone().detach().to(device).float()
    
    # 確保可求導
    images_unnorm.requires_grad = True
    
    # Forward
    outputs = model(normalize(images_unnorm, mean, std))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()
    
    # 取得梯度
    grad = images_unnorm.grad.data
    
    # 計算空間梯度幅度 (在通道維度上取平均)
    spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
    
    # 創建空間掩碼：選擇梯度最大的空間位置
    # 這裡選擇大於中位數的位置
    spatial_mask = (spatial_grad_magnitude > torch.median(spatial_grad_magnitude)).float()
    
    # 將空間掩碼擴展到所有通道
    expanded_spatial_mask = spatial_mask.repeat(1, images.size(1), 1, 1)
    
    # 應用掩碼到符號梯度
    grad_sign = torch.sign(grad)
    masked_grad_sign = grad_sign * expanded_spatial_mask
    
    # 生成對抗樣本
    adv_images_unnorm = images_unnorm + eps * masked_grad_sign
    
    # clamp 到 [0,1]
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)
    
    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def pgd_spatial_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std):
    """
    投影梯度下降(PGD)空間攻擊
    
    通過多步迭代優化，專注於攻擊重要的空間結構，對圖像的空間特徵進行擾動。
    
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 反正規化
    images_orig = unnormalize(images, mean, std).clone().detach().to(device)
    adv_images_unnorm = images_orig.clone().detach()
    
    # 保存空間掩碼（只計算一次）
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
            
            # 選擇梯度最大的前25%位置
            B = images.size(0)
            threshold = torch.quantile(spatial_grad_magnitude.view(B, -1), 0.75, dim=1).view(B, 1, 1, 1)
            spatial_mask = (spatial_grad_magnitude > threshold).float()
            
            # 擴展到所有通道
            spatial_mask = spatial_mask.repeat(1, images.size(1), 1, 1)
        
        # 應用掩碼
        masked_grad = grad * spatial_mask
        grad_sign = masked_grad.sign()
        
        # 更新對抗樣本
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        
        # 確保擾動在限制範圍內
        eta = torch.clamp(adv_images_unnorm - images_orig, min=-eps, max=eps)
        adv_images_unnorm = torch.clamp(images_orig + eta, 0, 1).detach()
    
    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def cw_spatial_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1,
                      device='cuda', mean=None, std=None):
    """
    Carlini & Wagner 空間攻擊
    
    通過優化損失函數，專注於攻擊重要的空間結構，生成高質量對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        c (float): 平衡係數
        kappa (float): 置信度參數
        steps (int): 優化步數
        lr (float): 學習率
        eps (float): 最大擾動範圍（限制L∞範數）
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 反正規化
    images_orig = unnormalize(images, mean, std).clone().detach().to(device)
    
    # 初始化空間擾動參數
    B, C, H, W = images_orig.shape
    
    # 我們只優化一個空間掩碼，對所有通道應用相同的擾動
    spatial_delta = torch.zeros((B, 1, H, W), requires_grad=True, device=device)
    
    # 優化器只針對空間擾動
    optimizer = torch.optim.Adam([spatial_delta], lr=lr)
    
    # 計算空間掩碼
    # 執行一次前向傳播來獲取初始梯度
    temp_images = images_orig.clone().detach()
    temp_images.requires_grad = True
    outputs = model(normalize(temp_images, mean, std))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad = temp_images.grad.data
    spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
    
    # 選擇梯度最大的前25%位置
    threshold = torch.quantile(spatial_grad_magnitude.view(B, -1), 0.75, dim=1).view(B, 1, 1, 1)
    spatial_mask = (spatial_grad_magnitude > threshold).float()
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分類任務
        target_onehot = torch.zeros(outputs.shape).to(device)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 找到目標類別之外的最大得分類別
        other = outputs.clone().detach()
        other[target_onehot > 0.5] = -float('inf')
        other_max = other.max(1, keepdim=True)[0]
        
        # 目標類別的得分
        target_score = torch.sum(outputs * target_onehot, 1)
        
        # 對抗損失：希望其他類別的得分大於目標類別的得分
        adv_loss = torch.clamp(target_score - other_max + kappa, min=0)
        return adv_loss.mean()
    
    # 優化過程
    for step in range(steps):
        optimizer.zero_grad()
        
        # 擴展空間擾動到所有通道，乘以空間掩碼
        expanded_delta = spatial_delta.repeat(1, C, 1, 1) * spatial_mask.repeat(1, C, 1, 1)
        
        # 生成當前對抗樣本
        adv_images_unnorm = torch.clamp(images_orig + expanded_delta, 0, 1)
        
        # 重新正規化後前向傳播
        outputs = model(normalize(adv_images_unnorm, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算CW損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 添加擾動大小的正則化項
        reg_loss = c * torch.norm(expanded_delta.view(B, -1), p=2, dim=1).mean()
        
        # 總損失
        total_loss = adv_loss + reg_loss
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
        
        # 確保擾動在eps範圍內
        with torch.no_grad():
            expanded_delta_clamped = torch.clamp(expanded_delta, min=-eps, max=eps)
            spatial_delta.data = torch.mean(
                expanded_delta_clamped.data, dim=1, keepdim=True
            ) * spatial_mask
    
    # 生成最終的對抗樣本
    with torch.no_grad():
        # 擴展空間擾動到所有通道，乘以空間掩碼
        expanded_delta = spatial_delta.repeat(1, C, 1, 1) * spatial_mask.repeat(1, C, 1, 1)
        
        # 確保擾動在eps範圍內
        expanded_delta = torch.clamp(expanded_delta, min=-eps, max=eps)
        
        # 生成最終對抗樣本
        adv_images_unnorm = torch.clamp(images_orig + expanded_delta, 0, 1)
        
        # 重新正規化
        adv_images = normalize(adv_images_unnorm, mean, std)
    
    return adv_images.detach()


def deepfool_spatial_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02,
                           device='cuda', mean=None, std=None):
    """
    DeepFool 空間攻擊
    
    通過迭代尋找最小擾動，使模型錯誤分類。專注於空間域的重要區域。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 類別數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過沖參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    B, C, H, W = images_unnorm.shape
    
    # 初始化空間掩碼（通過一次前向傳播獲取梯度）
    temp_images = images_unnorm.clone()
    temp_images.requires_grad = True
    
    # 前向傳播並獲取預測
    outputs = model(normalize(temp_images, mean, std))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # 獲取初始預測
    pred_orig = torch.argmax(outputs, dim=1)
    
    # 初始化結果
    adv_images_unnorm = images_unnorm.clone()
    
    # 為每個樣本計算對抗擾動
    for batch_idx in range(B):
        # 獲取當前樣本
        sample = images_unnorm[batch_idx:batch_idx+1].clone().detach().requires_grad_(True)
        
        # 獲取初始預測
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        f_0 = outputs[0, pred_orig[batch_idx]]
        
        # 計算梯度以獲取空間掩碼
        f_0.backward(retain_graph=True)
        grad_0 = sample.grad.data.clone()
        
        # 計算空間重要性掩碼
        spatial_importance = torch.mean(grad_0.abs(), dim=1, keepdim=True)
        threshold = torch.quantile(spatial_importance.view(-1), 0.75)
        spatial_mask = (spatial_importance > threshold).float()
        
        # 擴展到所有通道
        expanded_spatial_mask = spatial_mask.repeat(1, C, 1, 1)
        
        # 重置，準備正式的DeepFool迭代
        sample.grad.zero_()
        model.zero_grad()
        sample = images_unnorm[batch_idx:batch_idx+1].clone().detach().requires_grad_(True)
        
        current_pred = pred_orig[batch_idx]
        iteration = 0
        total_perturbation = torch.zeros_like(sample)
        
        while current_pred == pred_orig[batch_idx] and iteration < max_iter:
            outputs = model(normalize(sample, mean, std))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 初始化最小擾動
            w_direction = None
            f_direction = None
            min_distance = float('inf')
            
            # 檢查所有其他類別
            for k in range(num_classes):
                if k == current_pred:
                    continue
                
                zero_gradients(sample)
                
                # 計算類別k的得分
                f_k = outputs[0, k]
                f_k.backward(retain_graph=True)
                
                # 獲取梯度並應用空間掩碼
                grad_k = sample.grad.data.clone() * expanded_spatial_mask
                
                # 計算類別差異
                w_k = grad_k
                f_k = outputs[0, k] - outputs[0, current_pred]
                
                # 計算擾動方向和距離
                distance = abs(f_k) / (torch.norm(w_k.view(-1)) + 1e-8)
                
                # 更新最小擾動
                if distance < min_distance:
                    min_distance = distance
                    w_direction = w_k
                    f_direction = f_k
            
            # 沿最小擾動方向更新樣本
            if w_direction is not None:
                # 計算擾動大小
                pert_magnitude = abs(f_direction) / (torch.norm(w_direction.view(-1)) + 1e-8)
                
                # 計算擾動
                current_perturbation = (pert_magnitude + 1e-8) * w_direction / (torch.norm(w_direction) + 1e-8)
                
                # 更新總擾動
                total_perturbation += (1 + overshoot) * current_perturbation
                
                # 更新樣本
                sample = torch.clamp(images_unnorm[batch_idx:batch_idx+1] + total_perturbation, 0, 1)
                sample.requires_grad_(True)
                
                # 獲取新的預測
                outputs = model(normalize(sample, mean, std))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                current_pred = torch.argmax(outputs, dim=1)[0].item()
                
                iteration += 1
        
        # 更新批次中的對抗樣本
        adv_images_unnorm[batch_idx:batch_idx+1] = sample.detach()
    
    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach() 