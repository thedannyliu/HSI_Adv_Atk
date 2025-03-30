import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

def set_random_seed(seed):
    """
    設定隨機種子，確保實驗可重複性
    
    Args:
        seed (int): 隨機種子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, save_path, is_best=False):
    """
    保存模型檢查點
    
    Args:
        model: 模型
        optimizer: 優化器
        epoch (int): 當前訓練週期
        save_path (str): 保存路徑
        is_best (bool): 是否為最佳模型
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    # 保存最近一次權重
    latest_path = os.path.join(save_path, "latest.pth")
    torch.save(checkpoint, latest_path)
    
    # 如果是最佳結果，再額外保存一份
    if is_best:
        best_path = os.path.join(save_path, "best.pth")
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    載入模型檢查點
    
    Args:
        model: 模型
        optimizer: 優化器
        checkpoint_path (str): 檢查點路徑
        device: 計算設備
        
    Returns:
        int: 訓練週期
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} does not exist!")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    return epoch

def get_datetime_str():
    """
    獲取當前日期時間字符串
    
    Returns:
        str: 日期時間字符串，格式為 YYYYMMdd_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def visualize_results(image, mask, pred, save_path=None):
    """
    可視化分割結果
    
    Args:
        image (numpy.ndarray): 原始圖像，形狀為 [C, H, W]
        mask (numpy.ndarray): 真實標籤，形狀為 [H, W]
        pred (numpy.ndarray): 預測結果，形狀為 [H, W]
        save_path (str): 保存路徑，默認為None則顯示結果
    """
    # 取三個通道，用於RGB顯示
    if image.shape[0] > 3:
        rgb_image = image[:3].transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    else:
        rgb_image = image.transpose(1, 2, 0)
    
    # 歸一化到[0,1]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    
    # 創建可視化圖
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_adversarial(original, adversarial, original_pred, adversarial_pred, save_path=None):
    """
    可視化對抗樣本結果
    
    Args:
        original (numpy.ndarray): 原始圖像，形狀為 [C, H, W]
        adversarial (numpy.ndarray): 對抗樣本，形狀為 [C, H, W]
        original_pred (numpy.ndarray): 原始預測，形狀為 [H, W]
        adversarial_pred (numpy.ndarray): 對抗樣本預測，形狀為 [H, W]
        save_path (str): 保存路徑，默認為None則顯示結果
    """
    # 取三個通道，用於RGB顯示
    if original.shape[0] > 3:
        rgb_orig = original[:3].transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        rgb_adv = adversarial[:3].transpose(1, 2, 0)
    else:
        rgb_orig = original.transpose(1, 2, 0)
        rgb_adv = adversarial.transpose(1, 2, 0)
    
    # 歸一化到[0,1]
    rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)
    rgb_adv = (rgb_adv - rgb_adv.min()) / (rgb_adv.max() - rgb_adv.min() + 1e-8)
    
    # 計算差異圖
    diff = np.abs(rgb_adv - rgb_orig)
    diff = diff / diff.max()  # 歸一化差異
    
    # 創建可視化圖
    plt.figure(figsize=(15, 3))
    
    plt.subplot(1, 5, 1)
    plt.imshow(rgb_orig)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(original_pred, cmap='gray')
    plt.title('Original Prediction')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(rgb_adv)
    plt.title('Adversarial Image')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(adversarial_pred, cmap='gray')
    plt.title('Adversarial Prediction')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(diff)
    plt.title('Difference (x10)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 