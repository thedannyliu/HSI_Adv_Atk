import torch
import torch.nn.functional as F

def normalize(images, mean, std):
    """
    正規化圖像
    
    Args:
        images (torch.Tensor): 輸入圖像
        mean (list or float): 均值
        std (list or float): 標準差
        
    Returns:
        torch.Tensor: 正規化後的圖像
    """
    mean = torch.tensor(mean, dtype=torch.float).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std, dtype=torch.float).view(1, -1, 1, 1).to(images.device)
    images = (images - mean) / std
    return images

def unnormalize(images, mean, std):
    """
    反正規化圖像
    
    Args:
        images (torch.Tensor): 輸入圖像
        mean (list or float): 均值
        std (list or float): 標準差
        
    Returns:
        torch.Tensor: 反正規化後的圖像
    """
    mean = torch.tensor(mean, dtype=torch.float).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std, dtype=torch.float).view(1, -1, 1, 1).to(images.device)
    images = images * std + mean
    return images

def fgsm_attack_adaptive(model, images, labels, eps, criterion, device, mean, std):
    """
    自適應快速梯度符號法(FGSM)攻擊: 根據每個波段的梯度大小調整擾動
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        labels (torch.Tensor): 真實標籤
        eps (float): 擾動大小
        criterion: 損失函數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std)
    images_unnorm = images_unnorm.clone().detach().to(device).float()
    labels = labels.clone().detach().to(device)

    # 確保可求導
    images_unnorm.requires_grad = True

    # Forward
    outputs = model(normalize(images_unnorm, mean, std))
    if isinstance(outputs, tuple):
        # 若模型回傳 (pred, pred_bands)，取 pred 用來計算 loss
        outputs = outputs[0]
    loss = criterion(outputs, labels)
    
    # 反向傳播
    model.zero_grad()
    loss.backward()

    # 取得梯度
    grad = images_unnorm.grad.data

    # 計算每個波段的最大梯度
    grad_max = torch.max(torch.abs(grad), dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    grad_max = torch.where(grad_max == 0, torch.ones_like(grad_max), grad_max)  # 避免除以零

    # 計算自適應擾動
    grad_normalized = grad / grad_max
    adv_images_unnorm = images_unnorm + eps * grad_normalized.sign()

    # clamp 到 [0,1]
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)

    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()

def pgd_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std):
    """
    投影梯度下降(PGD)攻擊: x_{t+1} = Π_{||x - x_0||∞ ≤ eps} [ x_t + alpha * sign(grad_x) ]
    
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
    images_orig = unnormalize(images, mean, std).clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_images_unnorm = images_orig.clone().detach()

    for i in range(steps):
        adv_images_unnorm.requires_grad = True
        outputs = model(normalize(adv_images_unnorm, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        grad_sign = adv_images_unnorm.grad.data.sign()
        
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        eta = torch.clamp(adv_images_unnorm - images_orig, min=-eps, max=eps)
        adv_images_unnorm = torch.clamp(images_orig + eta, 0, 1).detach()

    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach() 