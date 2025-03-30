import torch
import numpy as np

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
            
def get_important_bands(hsi_data, n_bands=10):
    """
    根據高光譜資料的方差選擇重要波段
    
    Args:
        hsi_data (torch.Tensor): 高光譜資料，形狀為 [B, C, H, W]
        n_bands (int): 選擇的波段數量
        
    Returns:
        list: 重要波段的索引
    """
    if isinstance(hsi_data, torch.Tensor):
        hsi_data = hsi_data.detach().cpu().numpy()
        
    # 重塑為 [C, N] 形狀，其中 N = B*H*W
    C = hsi_data.shape[1]
    flattened = hsi_data.reshape(hsi_data.shape[0], C, -1)
    
    # 計算每個波段的方差
    variances = np.var(flattened, axis=2).mean(axis=0)
    
    # 選擇方差最大的 n_bands 個波段
    important_bands = np.argsort(variances)[-n_bands:]
    
    return important_bands.tolist() 