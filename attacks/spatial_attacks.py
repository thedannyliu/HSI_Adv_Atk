import torch
import torch.nn.functional as F
from .utils import normalize, unnormalize, zero_gradients

def fgsm_spatial_attack(model, images, labels, eps, criterion, device, mean, std):
    """
    空間域上的快速梯度符號法(FGSM)攻擊
    
    在空間域上進行攻擊時，專注於在空間位置上施加擾動，確保擾動在各波段間保持一致的空間模式。
    
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
    
    # 計算空間梯度幅度 (在通道維度上取平均)
    # 這樣確保對所有波段應用相同的空間位置擾動模式
    spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
    grad_sign = torch.sign(grad)
    
    # 應用空間一致性掩碼：將相同的空間位置掩碼應用到所有波段
    spatial_mask = (spatial_grad_magnitude > torch.median(spatial_grad_magnitude)).float()
    
    # 生成對抗樣本
    adv_images_unnorm = images_unnorm + eps * grad_sign * spatial_mask
    
    # clamp 到 [0,1]
    adv_images_unnorm = torch.clamp(adv_images_unnorm, 0, 1)

    # 重新正規化
    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def pgd_spatial_attack(model, images, labels, eps, alpha, steps, criterion, device, mean, std):
    """
    空間域上的投影梯度下降(PGD)攻擊
    
    在空間域上進行攻擊時，專注於在空間位置上施加擾動，確保擾動在各波段間保持一致的空間模式。
    
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
    
    B, C, H, W = images_orig.shape
    
    # 保存空間擾動掩碼
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
        
        if spatial_mask is None:
            # 首次迭代：基於原始梯度計算空間掩碼
            spatial_grad_magnitude = torch.mean(grad.abs(), dim=1, keepdim=True)
            
            # 只關注重要位置：選擇梯度最大的前25%像素位置
            threshold = torch.quantile(spatial_grad_magnitude.view(B, -1), 0.75, dim=1).view(B, 1, 1, 1)
            spatial_mask = (spatial_grad_magnitude > threshold).float()
            
            # 確保空間掩碼在通道維度上重複
            spatial_mask = spatial_mask.repeat(1, C, 1, 1)
        
        # 應用空間掩碼到梯度
        masked_grad = grad * spatial_mask
        grad_sign = masked_grad.sign()
        
        # 更新對抗樣本
        adv_images_unnorm = adv_images_unnorm + alpha * grad_sign
        
        # 確保擾動在限制範圍內
        eta = torch.clamp(adv_images_unnorm - images_orig, min=-eps, max=eps)
        adv_images_unnorm = torch.clamp(images_orig + eta, 0, 1).detach()

    adv_images = normalize(adv_images_unnorm, mean, std)
    return adv_images.detach()


def cw_spatial_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1, device='cuda', mean=None, std=None):
    """
    空間域上的Carlini & Wagner L2 攻擊
    
    在空間域上進行攻擊時，專注於在空間位置上施加擾動，確保擾動在各波段間保持一致的空間模式。
    
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
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    B, C, H, W = images_unnorm.shape
    
    # 初始化空間擾動（較少參數）：每個空間位置共享相同的擾動
    spatial_delta = torch.zeros((B, 1, H, W), requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([spatial_delta], lr=lr)
    
    # 用於計算對抗損失的函數
    def compute_adv_loss(outputs, targets):
        # 對於分割任務，我們希望模型的預測與真實標籤不同
        pred_labels = outputs.argmax(dim=1)
        return -F.nll_loss(F.log_softmax(outputs, dim=1), targets)

    for step in range(steps):
        optimizer.zero_grad()
        
        # 將空間擾動擴展到所有通道
        delta = spatial_delta.repeat(1, C, 1, 1)
        
        # 生成對抗樣本
        adv_images = torch.clamp(images_unnorm + delta, 0, 1)
        adv_images_norm = normalize(adv_images, mean, std)
        
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算對抗損失
        adv_loss = compute_adv_loss(outputs, labels)
        
        # 計算擾動的L2範數
        l2_loss = torch.sum(delta ** 2)
        
        # 總損失
        loss = adv_loss + c * l2_loss
        loss.backward()
        optimizer.step()
        
        # 限制擾動範圍在 [-eps, eps]
        spatial_delta.data = torch.clamp(spatial_delta.data, -eps, eps)
        
        if step % 100 == 0:
            print(f"[空間C&W] Step {step}/{steps}, Loss: {loss.item()}")
    
    # 將空間擾動擴展到所有通道
    final_delta = spatial_delta.repeat(1, C, 1, 1)
    
    adv_images = torch.clamp(images_unnorm + final_delta, 0, 1)
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach()


def deepfool_spatial_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02, device='cuda', mean=None, std=None):
    """
    空間域上的DeepFool攻擊
    
    在空間域上進行攻擊時，專注於在空間位置上施加擾動，確保擾動在各波段間保持一致的空間模式。
    
    Args:
        model (nn.Module): 目標模型
        images (torch.Tensor): 輸入圖像
        num_classes (int): 分類數量
        max_iter (int): 最大迭代次數
        overshoot (float): 過度參數
        device: 計算設備
        mean (list or float): 正規化均值
        std (list or float): 正規化標準差
        
    Returns:
        torch.Tensor: 對抗樣本
    """
    # 反正規化
    images_unnorm = unnormalize(images, mean, std).clone().detach().to(device)
    batch_size = images.shape[0]
    adv_images = images_unnorm.clone().detach()
    
    # 一次處理一個樣本
    for i in range(batch_size):
        sample = adv_images[i:i+1].clone().detach().requires_grad_(True)
        original_sample = images_unnorm[i:i+1].clone().detach()
        
        B, C, H, W = sample.shape
        
        outputs = model(normalize(sample, mean, std))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 獲取原始預測
        _, pred_orig = torch.max(outputs, 1)
        pred_orig = pred_orig.item()
        
        # 空間掩碼：記錄哪些空間位置的梯度最大
        spatial_mask = None
        
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
            
            # 尋找最近的決策邊界
            w_k = gradients[pred_orig]
            f_k = outputs[0, pred_orig]
            
            # 首次迭代：計算空間掩碼
            if spatial_mask is None and it == 0:
                # 計算每個空間位置的梯度幅度
                spatial_grad_magnitude = torch.mean(w_k.abs(), dim=1, keepdim=True)
                
                # 選擇梯度最大的前50%像素位置
                threshold = torch.quantile(spatial_grad_magnitude.view(B, -1), 0.5, dim=1).view(B, 1, 1, 1)
                spatial_mask = (spatial_grad_magnitude > threshold).float().repeat(1, C, 1, 1)
            
            min_dist = float('inf')
            closest_class = -1
            
            for k in range(num_classes):
                if k == pred_orig:
                    continue
                
                w_k_prime = gradients[k]
                f_k_prime = outputs[0, k]
                
                # 應用空間掩碼到梯度差
                if spatial_mask is not None:
                    w_diff = (w_k - w_k_prime) * spatial_mask
                else:
                    w_diff = w_k - w_k_prime
                    
                f_diff = f_k - f_k_prime
                
                # 計算到決策邊界的距離
                dist = abs(f_diff) / (w_diff.norm() + 1e-7)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_class = k
            
            # 如果找不到最近的類別，則停止
            if closest_class == -1:
                break
            
            # 計算擾動方向，應用空間掩碼
            if spatial_mask is not None:
                w_diff = (gradients[pred_orig] - gradients[closest_class]) * spatial_mask
            else:
                w_diff = gradients[pred_orig] - gradients[closest_class]
                
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