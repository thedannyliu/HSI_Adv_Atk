import torch
import torch.nn.functional as F

def normalize(images, mean=0.0, std=1.0):
    """
    正規化圖像
    Args:
        images: 輸入圖像 [B, C, H, W]
        mean: 均值
        std: 標準差
    Returns:
        torch.Tensor: 正規化後的圖像
    """
    # 確保輸入的mean和std是浮點數
    mean = float(mean)
    std = float(std)
    
    # 檢查std不為0
    if std < 1e-10:
        std = 1.0
        print("警告: 標準差接近0，已設為1.0")
    
    # 克隆圖像避免原地修改
    normalized = images.clone()
    
    # 應用正規化
    normalized = (normalized - mean) / std
    
    return normalized

def unnormalize(images, mean=0.0, std=1.0):
    """
    反正規化圖像
    Args:
        images: 輸入圖像 [B, C, H, W]
        mean: 均值
        std: 標準差
    Returns:
        torch.Tensor: 反正規化後的圖像
    """
    # 確保輸入的mean和std是浮點數
    mean = float(mean)
    std = float(std)
    
    # 檢查std不為0
    if std < 1e-10:
        std = 1.0
        print("警告: 標準差接近0，已設為1.0")
    
    # 克隆圖像避免原地修改
    unnormalized = images.clone()
    
    # 應用反正規化
    unnormalized = unnormalized * std + mean
    
    return unnormalized

def fgsm_attack(model, images, labels, eps, criterion, device, mean=None, std=None):
    """
    快速梯度符號法(FGSM)攻擊
    
    通過計算損失函數關於輸入的梯度，沿著梯度符號方向進行擾動，生成對抗樣本。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值，可選
        std (list or float): 正規化標準差，可選
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 處理正規化/反正規化
    if mean is not None and std is not None:
        # 反正規化圖像
        images_unnorm = unnormalize(images, mean, std)
        images_unnorm = images_unnorm.clone().detach().to(device).float()
        images_unnorm.requires_grad = True
        
        # 前向傳播 (用正規化數據)
        outputs = model(normalize(images_unnorm, mean, std))
    else:
        # 直接使用輸入
        images_t = images.clone().detach().to(device).float()
        images_t.requires_grad = True
        outputs = model(images_t)
    
    # 處理可能的多輸出情況
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # 計算損失
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()
    
    # 提取梯度並生成對抗樣本
    if mean is not None and std is not None:
        grad = images_unnorm.grad.data
        grad_sign = torch.sign(grad)
        adv_images_unnorm = images_unnorm + eps * grad_sign
        adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)
        adv_images = normalize(adv_images_unnorm, mean, std)
    else:
        grad = images_t.grad.data
        grad_sign = torch.sign(grad)
        adv_images = images_t + eps * grad_sign
        adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()

def fgsm_attack_adaptive(model, images, labels, eps, criterion, device, mean=None, std=None):
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 處理正規化/反正規化
    if mean is not None and std is not None:
        # 反正規化圖像
        images_unnorm = unnormalize(images, mean, std)
        images_unnorm = images_unnorm.clone().detach().to(device).float()
        images_unnorm.requires_grad = True
        
        # 前向傳播 (用正規化數據)
        outputs = model(normalize(images_unnorm, mean, std))
    else:
        # 直接使用輸入
        images_t = images.clone().detach().to(device).float()
        images_t.requires_grad = True
        outputs = model(images_t)
    
    # 處理可能的多輸出情況
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    
    # 計算損失
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()
    
    # 提取梯度並生成對抗樣本
    if mean is not None and std is not None:
        grad = images_unnorm.grad.data
        
        # 自適應調整：計算每個通道的梯度重要性
        channel_importance = torch.mean(grad.abs(), dim=(2, 3), keepdim=True)
        # 歸一化以保持總擾動大小不變
        channel_importance = channel_importance / (torch.sum(channel_importance) + 1e-8)
        
        # 應用通道重要性到梯度
        weighted_grad = grad * channel_importance
        grad_sign = torch.sign(weighted_grad)
        
        adv_images_unnorm = images_unnorm + eps * grad_sign
        adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)
        adv_images = normalize(adv_images_unnorm, mean, std)
    else:
        grad = images_t.grad.data
        
        # 自適應調整
        channel_importance = torch.mean(grad.abs(), dim=(2, 3), keepdim=True)
        channel_importance = channel_importance / (torch.sum(channel_importance) + 1e-8)
        weighted_grad = grad * channel_importance
        grad_sign = torch.sign(weighted_grad)
        
        adv_images = images_t + eps * grad_sign
        adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()

def pgd_attack(model, images, labels, eps, alpha, steps, criterion, device, mean=None, std=None):
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 確保標籤是 Long 類型
    labels = labels.clone().detach().to(device).long()
    
    # 處理正規化/反正規化
    if mean is not None and std is not None:
        # 反正規化原始圖像用於邊界檢查
        images_orig = unnormalize(images, mean, std).clone().detach().to(device)
        adv_images = images_orig.clone().detach()
    else:
        images_orig = images.clone().detach().to(device)
        adv_images = images_orig.clone().detach()
    
    for i in range(steps):
        adv_images.requires_grad = True
        
        # 前向傳播
        if mean is not None and std is not None:
            outputs = model(normalize(adv_images, mean, std))
        else:
            outputs = model(adv_images)
        
        # 處理可能的多輸出情況
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算損失
        loss = criterion(outputs, labels)
        
        # 反向傳播
        model.zero_grad()
        loss.backward()
        
        # 更新對抗樣本
        grad = adv_images.grad.data
        adv_images = adv_images.detach() + alpha * torch.sign(grad)
        
        # 確保擾動在epsilon範圍內
        eta = torch.clamp(adv_images - images_orig, -eps, eps)
        adv_images = torch.clamp(images_orig + eta, 0, 1).detach()
    
    # 如果需要，重新正規化
    if mean is not None and std is not None:
        adv_images = normalize(adv_images, mean, std)
    
    return adv_images 