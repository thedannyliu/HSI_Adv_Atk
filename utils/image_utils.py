import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def save_image(image, save_path):
    """
    保存圖像
    
    Args:
        image (numpy.ndarray or torch.Tensor): 要保存的圖像，形狀為 [C, H, W]
        save_path (str): 保存路徑
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 轉換為 numpy 數組
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 確保圖像在 [0, 1] 範圍內
    image = np.clip(image, 0, 1)
    
    # 如果是多通道圖像，轉換為 RGB
    if image.shape[0] > 3:
        # 取前三個通道
        image = image[:3]
    
    # 轉換為 PIL 圖像並保存
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    Image.fromarray(image).save(save_path)

def save_rgb(image, save_path):
    """
    保存 RGB 圖像
    
    Args:
        image (numpy.ndarray or torch.Tensor): 要保存的圖像，形狀為 [C, H, W]
        save_path (str): 保存路徑
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 轉換為 numpy 數組
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 確保圖像在 [0, 1] 範圍內
    image = np.clip(image, 0, 1)
    
    # 如果是多通道圖像，取前三個通道
    if image.shape[0] > 3:
        image = image[:3]
    
    # 轉換為 PIL 圖像並保存
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    Image.fromarray(image).save(save_path)

def save_rgb_comparison(original, adversarial, save_path):
    """
    保存原始圖像和對抗樣本的比較圖
    
    Args:
        original (numpy.ndarray or torch.Tensor): 原始圖像，形狀為 [C, H, W]
        adversarial (numpy.ndarray or torch.Tensor): 對抗樣本，形狀為 [C, H, W]
        save_path (str): 保存路徑
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 轉換為 numpy 數組
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(adversarial, torch.Tensor):
        adversarial = adversarial.detach().cpu().numpy()
    
    # 確保圖像在 [0, 1] 範圍內
    original = np.clip(original, 0, 1)
    adversarial = np.clip(adversarial, 0, 1)
    
    # 如果是多通道圖像，取前三個通道
    if original.shape[0] > 3:
        original = original[:3]
    if adversarial.shape[0] > 3:
        adversarial = adversarial[:3]
    
    # 計算差異
    diff = np.abs(adversarial - original)
    diff = np.clip(diff * 10, 0, 1)  # 放大差異以便觀察
    
    # 創建比較圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始圖像
    axes[0].imshow(np.transpose(original, (1, 2, 0)))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 對抗樣本
    axes[1].imshow(np.transpose(adversarial, (1, 2, 0)))
    axes[1].set_title('Adversarial')
    axes[1].axis('off')
    
    # 差異圖
    axes[2].imshow(np.transpose(diff, (1, 2, 0)))
    axes[2].set_title('Difference (x10)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_attack_comparison(clean_img, adv_img, save_path, magnification=50, title=None):
    """
    創建一個包含三張圖的可視化：原始圖像、對抗樣本和差異（放大）
    
    Args:
        clean_img (numpy.ndarray or torch.Tensor): 原始圖像，形狀為 [C, H, W]
        adv_img (numpy.ndarray or torch.Tensor): 對抗樣本，形狀為 [C, H, W]
        save_path (str): 保存路徑
        magnification (int): 差異放大倍數，預設為50
        title (str, optional): 圖像標題
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 轉換為 numpy 數組
    if isinstance(clean_img, torch.Tensor):
        clean_img = clean_img.detach().cpu().numpy()
    if isinstance(adv_img, torch.Tensor):
        adv_img = adv_img.detach().cpu().numpy()
    
    # 確保圖像在 [0, 1] 範圍內
    clean_img = np.clip(clean_img, 0, 1)
    adv_img = np.clip(adv_img, 0, 1)
    
    # 如果是多通道圖像，取前三個通道作為RGB
    if clean_img.shape[0] > 3:
        clean_img = clean_img[:3]
    if adv_img.shape[0] > 3:
        adv_img = adv_img[:3]
    
    # 計算差異並放大
    diff = np.abs(adv_img - clean_img) * magnification
    diff = np.clip(diff, 0, 1)  # 確保在 [0, 1] 範圍內
    
    # 創建子圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始圖像
    axes[0].imshow(np.transpose(clean_img, (1, 2, 0)))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 對抗樣本
    axes[1].imshow(np.transpose(adv_img, (1, 2, 0)))
    axes[1].set_title('Adversarial')
    axes[1].axis('off')
    
    # 差異圖
    axes[2].imshow(np.transpose(diff, (1, 2, 0)))
    axes[2].set_title(f'Difference (x{magnification})')
    axes[2].axis('off')
    
    # 設置總標題
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() 