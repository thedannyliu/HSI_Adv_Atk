import torch
import torch.nn.functional as F
from .utils import normalize, unnormalize, zero_gradients, get_important_bands

def fgsm_spectral_attack(model, images, labels, eps, criterion, device, mean, std, target_bands=None):
    """
    光譜域上的快速梯度符號法(FGSM)攻擊
    
    在光譜域上進行攻擊時，專注於修改特定波段的光譜特徵，保持空間結構相對不變。
    
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std)
    images_unnorm = images_unnorm.clone().detach().to(device).float()
    labels = labels.clone().detach().to(device)

    B, C, H, W = images_unnorm.shape
    
    # 如果未指定目標波段，則自動選擇重要波段
    if target_bands is None:
        target_bands = get_important_bands(images_unnorm, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))

    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0

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
    
    # 應用光譜掩碼
    grad = grad * spectral_mask
    grad_sign = torch.sign(grad)
    
    # 生成對抗樣本
    adv_images_unnorm = images_unnorm + eps * grad_sign
    
    # clamp 到 [0,1]
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)

    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def pgd_spectral_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std, target_bands=None):
    """
    光譜域上的投影梯度下降(PGD)攻擊
    
    在光譜域上進行攻擊時，專注於修改特定波段的光譜特徵，保持空間結構相對不變。
    
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    images_orig = unnormalize(images, mean, std).clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv_images_unnorm = images_orig.clone().detach()
    
    B, C, H, W = images_orig.shape
    
    # 如果未指定目標波段，則自動選擇重要波段
    if target_bands is None:
        target_bands = get_important_bands(images_orig, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))

    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0

    for i in range(steps):
        adv_images_unnorm.requires_grad = True
        outputs = model(normalize(adv_images_unnorm, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # 應用光譜掩碼到梯度
        grad = adv_images_unnorm.grad.data * spectral_mask
        grad_sign = grad.sign()
        
        # 更新對抗樣本
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        
        # 確保擾動在限制範圍內且僅應用於目標波段
        delta = adv_images_unnorm - images_orig
        delta = delta * spectral_mask  # 只在選定的波段上應用擾動
        delta = torch.clamp(delta, -eps, eps)
        adv_images_unnorm = torch.clamp(images_orig + delta, 0, 1).detach()

    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def cw_spectral_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1, device='cuda', mean=None, std=None, target_bands=None):
    """
    光譜域上的Carlini & Wagner L2 攻擊
    
    在光譜域上進行攻擊時，專注於修改特定波段的光譜特徵，保持空間結構相對不變。
    
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    B, C, H, W = images_unnorm.shape
    
    # 如果未指定目標波段，則自動選擇重要波段
    if target_bands is None:
        target_bands = get_important_bands(images_unnorm, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))
    
    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros(C, device=device)
    for band_idx in target_bands:
        spectral_mask[band_idx] = 1.0
    
    # 初始化光譜擾動（只對選定的波段）
    delta = torch.zeros_like(images_unnorm, requires_grad=True, device=device)
    
    # 只優化選定波段的擾動
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分割任務，我們希望模型的預測與真實標籤不同
        pred_labels = outputs.argmax(dim=1)
        return -F.nll_loss(F.log_softmax(outputs, dim=1), targets)

    for step in range(steps):
        optimizer.zero_grad()
        
        # 應用光譜掩碼：只修改選定的波段
        masked_delta = delta * spectral_mask.view(1, -1, 1, 1)
        
        # 生成對抗樣本
        adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)
        adv_images_norm = normalize(adv_images, mean, std)
        
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 計算擾動的L2範數（只考慮選定波段）
        l2_loss = torch.sum(masked_delta ** 2)
        
        # 總損失
        loss = adv_loss + c * l2_loss
        loss.backward()
        optimizer.step()
        
        # 限制擾動範圍在 [-eps, eps]，只應用於選定波段
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
        
        if step % 100 == 0:
            print(f"[光譜C&W] Step {step}/{steps}, Loss: {loss.item()}")
    
    # 應用最終的光譜擾動
    masked_delta = delta * spectral_mask.view(1, -1, 1, 1)
    adv_images = torch.clamp(images_unnorm + masked_delta, 0, 1)
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach()


def deepfool_spectral_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02, device='cuda', mean=None, std=None, target_bands=None):
    """
    光譜域上的DeepFool攻擊
    
    在光譜域上進行攻擊時，專注於修改特定波段的光譜特徵，保持空間結構相對不變。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 分類數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過度參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        target_bands (list): 要攻擊的目標波段，如果為None，會自動選擇重要波段
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    batch_size = images.shape[0]
    adv_images = images_unnorm.clone().detach()
    
    B, C, H, W = images_unnorm.shape
    
    # 如果未指定目標波段，則自動選擇重要波段
    if target_bands is None:
        target_bands = get_important_bands(images_unnorm, n_bands=C//3)
    
    # 確保目標波段列表不為空
    if not target_bands:
        target_bands = list(range(C))
    
    # 創建光譜掩碼：只攻擊特定波段
    spectral_mask = torch.zeros((1, C, 1, 1), device=device)
    for band_idx in target_bands:
        spectral_mask[0, band_idx, 0, 0] = 1.0
    
    # 一次處理一個樣本
    for i in range(batch_size):
        sample = adv_images[i:i+1].clone().detach().requires_grad_(True)
        original_sample = images_unnorm[i:i+1].clone().detach()
        
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 獲取原始預測
        _, pred_orig = torch.max(outputs, 1)
        pred_orig = pred_orig.item()
        
        # 迭代尋找最小擾動
        for it in range(max_iter):
            # 計算梯度
            gradients = []
            for k in range(num_classes):
                zero_gradients(sample)
                outputs = model(normalize(sample, mean, std))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 計算對第k類的梯度
                class_score = outputs[0, k]
                class_score.backward(retain_graph=True)
                grad = sample.grad.data.clone()
                
                # 應用光譜掩碼
                grad = grad * spectral_mask
                
                gradients.append(grad)
            
            # 如果已經成功誤導，則停止
            outputs = model(normalize(sample, mean, std))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, current_pred = torch.max(outputs, 1)
            if current_pred.item() != pred_orig and it > 0:
                break
            
            # 尋找最近的決策邊界
            w_k = gradients[pred_orig]
            f_k = outputs[0, pred_orig]
            
            min_dist = float('inf')
            closest_class = -1
            
            for k in range(num_classes):
                if k == pred_orig:
                    continue
                
                w_k_prime = gradients[k]
                f_k_prime = outputs[0, k]
                
                # 計算到決策邊界的距離
                w_diff = w_k - w_k_prime
                f_diff = f_k - f_k_prime
                
                dist = abs(f_diff) / (w_diff.norm() + 1e-7)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
            
            # 如果找不到最近的類別，則停止
            if closest_class == -1:
                break
            
            # 計算擾動方向
            w_diff = gradients[pred_orig] - gradients[closest_class]
            f_diff = outputs[0, pred_orig] - outputs[0, closest_class]
            
            # 計算擾動大小
            pert_magnitude = abs(f_diff) / (w_diff.norm() + 1e-7)
            
            # 添加擾動
            perturbation = (1 + overshoot) * pert_magnitude * w_diff / (w_diff.norm() + 1e-7)
            sample = sample + perturbation
            
            # 確保樣本在合理範圍內
            sample = torch.clamp(sample, 0, 1).detach().requires_grad_(True)
        
        adv_images[i:i+1] = sample
    
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach() 