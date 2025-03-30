import torch
import torch.nn.functional as F
from .utils import normalize, unnormalize, zero_gradients, get_important_bands

def fgsm_hybrid_attack(model, images, labels, eps, criterion, device, mean, std, 
                     spatial_weight=0.5, target_bands=None):
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std)
    images_unnorm = images_unnorm.clone().detach().to(device).float()
    labels = labels.clone().detach().to(device)

    B, C, H, W = images_unnorm.shape
    
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
    
    # 步驟1: 計算空間域部分
    # 計算空間梯度幅度 (在通道維度上取平均)
    spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
    
    # 創建空間掩碼：選擇梯度最大的空間位置
    spatial_mask = (spatial_grad_magnitude > torch.median(spatial_grad_magnitude)).float()
    
    # 步驟2: 計算光譜域部分
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
    
    # 生成對抗樣本
    adv_images_unnorm = images_unnorm + eps * masked_grad_sign
    
    # clamp 到 [0,1]
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)

    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def pgd_hybrid_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std,
                     spatial_weight=0.5, target_bands=None):
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
        eta = torch.clamp(adv_images_unnorm - images_orig, min=-eps, max=eps)
        adv_images_unnorm = torch.clamp(images_orig + eta, 0, 1).detach()

    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def cw_hybrid_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1,
                    device='cuda', mean=None, std=None, spatial_weight=0.5, target_bands=None):
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
        
        # 2. 計算光譜擾動部分（只在選定波段上有擾動）
        masked_spectral_delta = spectral_delta * spectral_mask
        
        # 3. 合併兩種擾動（加權）
        hybrid_delta = spatial_weight * expanded_spatial_delta + (1 - spatial_weight) * masked_spectral_delta
        
        # 生成對抗樣本
        adv_images = torch.clamp(images_unnorm + hybrid_delta, 0, 1)
        adv_images_norm = normalize(adv_images, mean, std)
        
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 計算擾動的L2範數
        # 分別計算兩種擾動的L2範數並按權重加權
        spatial_l2 = torch.sum(expanded_spatial_delta ** 2)
        spectral_l2 = torch.sum(masked_spectral_delta ** 2)
        l2_loss = spatial_weight * spatial_l2 + (1 - spatial_weight) * spectral_l2
        
        # 總損失
        loss = adv_loss + c * l2_loss
        loss.backward()
        optimizer.step()
        
        # 限制擾動範圍在 [-eps, eps]
        with torch.no_grad():
            spatial_delta.data = torch.clamp(spatial_delta.data, -eps, eps)
            spectral_delta.data = torch.clamp(spectral_delta.data, -eps, eps)
        
        if step % 100 == 0:
            print(f"[混合C&W] Step {step}/{steps}, Loss: {loss.item()}")
    
    # 計算最終混合擾動
    expanded_spatial_delta = spatial_delta.repeat(1, C, 1, 1)
    masked_spectral_delta = spectral_delta * spectral_mask
    hybrid_delta = spatial_weight * expanded_spatial_delta + (1 - spatial_weight) * masked_spectral_delta
    
    # 生成對抗樣本
    adv_images = torch.clamp(images_unnorm + hybrid_delta, 0, 1)
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach()


def deepfool_hybrid_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02,
                           device='cuda', mean=None, std=None, spatial_weight=0.5, target_bands=None):
    """
    結合空間域和光譜域的混合DeepFool攻擊
    
    尋找空間和光譜域上的最小擾動，使模型預測結果改變。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 分類數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過度參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        spatial_weight (float): 空間域攻擊權重，範圍[0,1]，預設0.5表示均等混合
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
        
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 獲取原始預測
        _, pred_orig = torch.max(outputs, 1)
        pred_orig = pred_orig.item()
        
        # 空間掩碼和光譜掩碼
        sample_spatial_mask = None
        
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
                gradients.append(grad)
            
            # 如果已經成功誤導，則停止
            outputs = model(normalize(sample, mean, std))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, current_pred = torch.max(outputs, 1)
            if current_pred.item() != pred_orig and it > 0:
                break
            
            # 首次迭代：計算空間掩碼
            if sample_spatial_mask is None and it == 0:
                # 使用預測類別的梯度計算空間重要性
                w_k = gradients[pred_orig]
                spatial_grad_magnitude = torch.mean(w_k.abs(), dim=1, keepdim=True)
                
                # 選擇梯度最大的前40%像素位置
                threshold = torch.quantile(spatial_grad_magnitude.view(1, -1), 0.6, dim=1).view(1, 1, 1, 1)
                sample_spatial_mask = (spatial_grad_magnitude > threshold).float().repeat(1, C, 1, 1)
            
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
                
                # 應用混合掩碼到梯度差
                # 結合空間掩碼和光譜掩碼
                hybrid_mask = torch.max(
                    spatial_weight * sample_spatial_mask,
                    (1 - spatial_weight) * spectral_mask
                )
                
                w_diff = (w_k - w_k_prime) * hybrid_mask
                f_diff = f_k - f_k_prime
                
                # 計算到決策邊界的距離
                dist = abs(f_diff) / (w_diff.norm() + 1e-7)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
            
            # 如果找不到最近的類別，則停止
            if closest_class == -1:
                break
            
            # 計算擾動方向，應用混合掩碼
            hybrid_mask = torch.max(
                spatial_weight * sample_spatial_mask,
                (1 - spatial_weight) * spectral_mask
            )
            
            w_diff = (gradients[pred_orig] - gradients[closest_class]) * hybrid_mask
            f_diff = outputs[0, pred_orig] - outputs[0, closest_class]
            
            # 計算擾動大小
            pert_magnitude = abs(f_diff) / (w_diff.norm() + 1e-7)
            
            # 添加擾動
            perturbation = (1 + overshoot) * pert_magnitude * w_diff / (w_diff.norm() + 1e-7)
            sample = sample + perturbation
            sample = torch.clamp(sample, 0, 1).detach().requires_grad_(True)
        
        adv_images[i:i+1] = sample
    
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach() 
 
 