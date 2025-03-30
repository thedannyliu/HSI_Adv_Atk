import torch
import torch.nn.functional as F
from .basic_attacks import normalize, unnormalize

def cw_attack(model, images, labels, c=0.01, kappa=0, steps=1000, lr=0.01, eps=0.1, device='cuda', mean=None, std=None):
    """
    Carlini & Wagner L2 攻擊
    
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

    # 初始化擾動
    delta = torch.zeros_like(images_unnorm, requires_grad=True).to(device)

    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        adv_images = torch.clamp(images_unnorm + delta, 0, 1)
        adv_images_norm = normalize(adv_images, mean, std)
        outputs = model(adv_images_norm)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 計算分類損失
        loss1 = F.cross_entropy(outputs, labels)
        
        # 計算擾動的L2範數
        loss2 = torch.sum((adv_images - images_unnorm) ** 2)
        
        # 總損失
        loss = loss1 + c * loss2
        loss.backward()
        optimizer.step()

        # 限制擾動範圍在 [-eps, eps]
        delta.data = torch.clamp(delta, -eps, eps)

        if step % 100 == 0:
            print(f"[C&W] Step {step}/{steps}, Loss: {loss.item()}")

    adv_images = torch.clamp(images_unnorm + delta, 0, 1)
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach()

def deepfool_attack(model, images, num_classes=2, max_iter=50, overshoot=0.02, device='cuda', mean=None, std=None):
    """
    DeepFool 攻擊
    
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
            sample = torch.clamp(sample, 0, 1).detach().requires_grad_(True)
        
        adv_images[i:i+1] = sample
    
    adv_images_norm = normalize(adv_images, mean, std)
    return adv_images_norm.detach()

def zero_gradients(x):
    """
    將梯度設為零
    
    Args:
        x (torch.Tensor): 需要清零梯度的張量
    """
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, list) or isinstance(x, tuple):
        for elem in x:
            zero_gradients(elem) 