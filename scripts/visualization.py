import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import cv2
from sklearn.decomposition import PCA

def visualize_adversarial(orig_image, adv_image, label=None, pred_orig=None, pred_adv=None, output_path=None, mode='test', args=None):
    """
    可視化對抗樣本
    Args:
        orig_image (torch.Tensor/numpy.ndarray): 原始圖像 (C, H, W)
        adv_image (torch.Tensor/numpy.ndarray): 對抗樣本 (C, H, W)
        label (torch.Tensor/numpy.ndarray, optional): 真實標籤 (H, W)
        pred_orig (torch.Tensor/numpy.ndarray, optional): 原始預測結果 (H, W)
        pred_adv (torch.Tensor/numpy.ndarray, optional): 對抗預測結果 (H, W)
        output_path (str): 輸出路徑
        mode (str): 模式（'test' 或 'train'）
        args (argparse.Namespace): 命令行參數
    """
    # 檢查參數類型，確保數據格式正確
    if isinstance(orig_image, torch.Tensor):
        orig_image = orig_image.detach().cpu().numpy()
    if isinstance(adv_image, torch.Tensor):
        adv_image = adv_image.detach().cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if pred_orig is not None and isinstance(pred_orig, torch.Tensor):
        pred_orig = pred_orig.detach().cpu().numpy()
    if pred_adv is not None and isinstance(pred_adv, torch.Tensor):
        pred_adv = pred_adv.detach().cpu().numpy()

    # 創建輸出目錄
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 計算擾動
    perturbation = adv_image - orig_image
    
    # 使用方差最大的三個通道進行可視化
    variances = np.var(orig_image, axis=(1, 2))
    top_channels = np.argsort(variances)[-3:]
    
    # 創建RGB顯示圖像
    rgb_orig = np.zeros((orig_image.shape[1], orig_image.shape[2], 3))
    rgb_adv = np.zeros((adv_image.shape[1], adv_image.shape[2], 3))
    
    for j, channel in enumerate(top_channels):
        rgb_orig[..., j] = orig_image[channel]
        rgb_adv[..., j] = adv_image[channel]
    
    # 規範化到0-1範圍
    rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)
    rgb_adv = (rgb_adv - rgb_adv.min()) / (rgb_adv.max() - rgb_adv.min() + 1e-8)
    
    # 計算絕對擾動可視化
    rgb_pert = np.abs(rgb_adv - rgb_orig)
    # 增強擾動顯示
    enhanced_factor = 5.0
    rgb_pert = np.clip(rgb_pert * enhanced_factor, 0, 1)
    
    # 創建分析組合圖
    if label is not None and pred_orig is not None and pred_adv is not None:
        plt.figure(figsize=(18, 10))
        
        # 1. 原始圖像
        plt.subplot(2, 3, 1)
        plt.imshow(rgb_orig)
        plt.title('原始圖像')
        plt.axis('off')
        
        # 2. 對抗圖像
        plt.subplot(2, 3, 2)
        plt.imshow(rgb_adv)
        plt.title('對抗圖像')
        plt.axis('off')
        
        # 3. 擾動可視化（增強版）
        plt.subplot(2, 3, 3)
        plt.imshow(rgb_pert)
        plt.title(f'擾動 (x{enhanced_factor:.1f})')
        plt.axis('off')
        
        # 4. 真實標籤
        plt.subplot(2, 3, 4)
        plt.imshow(label, cmap='jet')
        plt.title('真實標籤')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # 5. 原始預測結果
        plt.subplot(2, 3, 5)
        plt.imshow(pred_orig, cmap='jet')
        plt.title('原始預測')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # 6. 對抗預測結果
        plt.subplot(2, 3, 6)
        plt.imshow(pred_adv, cmap='jet')
        plt.title('對抗預測')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        # 簡化版顯示（僅顯示圖像和擾動）
        plt.figure(figsize=(15, 5))
        
        # 1. 原始圖像
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_orig)
        plt.title('原始圖像')
        plt.axis('off')
        
        # 2. 對抗圖像
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_adv)
        plt.title('對抗圖像')
        plt.axis('off')
        
        # 3. 擾動可視化（增強版）
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_pert)
        plt.title(f'擾動 (x{enhanced_factor:.1f})')
        plt.axis('off')
    
    # 保存圖像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存詳細通道分析（如果有詳細模式）
    if args and getattr(args, "detailed_visualization", False):
        # 創建通道分析目錄
        channel_dir = os.path.join(os.path.dirname(output_path), "channels")
        os.makedirs(channel_dir, exist_ok=True)
        
        # 分析最多5個主要通道
        for c in range(min(5, orig_image.shape[0])):
            plt.figure(figsize=(15, 5))
            
            # 原始通道
            plt.subplot(1, 3, 1)
            plt.imshow(orig_image[c], cmap='viridis')
            plt.title(f'原始通道 {c}')
            plt.colorbar()
            
            # 對抗通道
            plt.subplot(1, 3, 2)
            plt.imshow(adv_image[c], cmap='viridis')
            plt.title(f'對抗通道 {c}')
            plt.colorbar()
            
            # 擾動
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(adv_image[c] - orig_image[c]), cmap='hot')
            plt.title(f'通道 {c} 擾動')
            plt.colorbar()
            
            # 保存通道分析圖
            channel_path = os.path.join(channel_dir, f"{os.path.basename(output_path).split('.')[0]}_channel_{c}.png")
            plt.tight_layout()
            plt.savefig(channel_path, dpi=200)
            plt.close()

def visualize_components(image, component_dim=3, output_path=None):
    """
    使用PCA將高維圖像可視化為RGB圖像
    Args:
        image (numpy.ndarray): 高維圖像 (C, H, W)
        component_dim (int): 要減少到的維度
        output_path (str): 輸出路徑
    """
    # 檢查輸入是否是torch.Tensor
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 獲取圖像尺寸
    c, h, w = image.shape
    
    # 將圖像重塑為形狀(C, H*W)
    image_reshaped = image.reshape(c, -1).T
    
    # 應用PCA進行降維
    pca = PCA(n_components=component_dim)
    components = pca.fit_transform(image_reshaped)
    
    # 將結果重塑回(H, W, component_dim)
    components = components.reshape(h, w, component_dim)
    
    # 規範化到0-1範圍
    for i in range(component_dim):
        min_val = components[..., i].min()
        max_val = components[..., i].max()
        if max_val > min_val:
            components[..., i] = (components[..., i] - min_val) / (max_val - min_val)
    
    # 如果要求3個分量，則創建RGB圖像
    if component_dim == 3:
        rgb_image = components
    else:
        # 否則只取前三個分量
        rgb_image = np.zeros((h, w, 3))
        for i in range(min(component_dim, 3)):
            rgb_image[..., i] = components[..., i]
    
    # 顯示和保存
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(f'PCA Components ({component_dim} components)')
    plt.axis('off')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return rgb_image

def visualize_hyperspectral(hyperspectral_image, method='pca', bands=None, output_path=None):
    """
    可視化高光譜圖像
    Args:
        hyperspectral_image (numpy.ndarray): 高光譜圖像 (C, H, W)
        method (str): 可視化方法 ('pca', 'selected_bands')
        bands (list): 要選擇的波段索引，僅當method='selected_bands'時使用
        output_path (str): 輸出路徑
    Returns:
        numpy.ndarray: 可視化後的RGB圖像
    """
    if isinstance(hyperspectral_image, torch.Tensor):
        hyperspectral_image = hyperspectral_image.detach().cpu().numpy()
    
    if method == 'pca':
        return visualize_components(hyperspectral_image, component_dim=3, output_path=output_path)
    
    elif method == 'selected_bands':
        if bands is None or len(bands) < 3:
            # 默認使用方差最大的三個波段
            variances = np.var(hyperspectral_image, axis=(1, 2))
            bands = np.argsort(variances)[-3:]
        
        # 選擇指定的三個波段
        rgb_image = np.zeros((hyperspectral_image.shape[1], hyperspectral_image.shape[2], 3))
        
        for i, band in enumerate(bands[:3]):
            band_data = hyperspectral_image[band]
            min_val = band_data.min()
            max_val = band_data.max()
            if max_val > min_val:
                rgb_image[..., i] = (band_data - min_val) / (max_val - min_val)
        
        # 顯示和保存
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title(f'Selected Bands: {bands[:3]}')
        plt.axis('off')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return rgb_image
    
    else:
        raise ValueError(f"Unsupported visualization method: {method}")

def visualize_prediction(pred, label=None, num_classes=None, output_path=None, colormap='jet'):
    """
    可視化分割預測結果
    Args:
        pred (torch.Tensor/numpy.ndarray): 預測結果 (H, W)
        label (torch.Tensor/numpy.ndarray, optional): 真實標籤 (H, W)
        num_classes (int, optional): 類別數量
        output_path (str): 輸出路徑
        colormap (str): 使用的顏色映射
    """
    # 檢查輸入是否是torch.Tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    
    # 確定類別數量
    if num_classes is None:
        num_classes = max(np.max(pred) + 1, np.max(label) + 1 if label is not None else 0)
    
    # 創建圖像
    if label is not None:
        plt.figure(figsize=(12, 6))
        
        # 顯示真實標籤
        plt.subplot(1, 2, 1)
        plt.imshow(label, cmap=colormap, vmin=0, vmax=num_classes-1)
        plt.title('Ground Truth')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 顯示預測結果
        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap=colormap, vmin=0, vmax=num_classes-1)
        plt.title('Prediction')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(pred, cmap=colormap, vmin=0, vmax=num_classes-1)
        plt.title('Prediction')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    # 保存圖像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close() 