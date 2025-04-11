import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torchvision.utils import make_grid


def visualize_hyperspectral_channel(hsi, channel=None, ax=None, title=None, cmap='viridis'):
    """
    可視化高光譜影像的單一通道
    
    Args:
        hsi: 高光譜影像 [C, H, W]
        channel: 要可視化的通道索引, 如果為None, 則使用中間通道
        ax: matplotlib 軸對象
        title: 圖的標題
        cmap: 色彩映射
        
    Returns:
        ax: matplotlib 軸對象
    """
    C, H, W = hsi.shape
    
    if channel is None:
        channel = C // 2  # 使用中間通道
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    band = hsi[channel]
    vmin, vmax = np.percentile(band, (2, 98))  # 增強對比度
    im = ax.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Channel {channel}')
    
    plt.colorbar(im, ax=ax)
    
    return ax


def visualize_rgb_from_hsi(hsi, r_band=29, g_band=19, b_band=9, ax=None, title=None):
    """
    從高光譜影像建立RGB假彩色影像
    
    Args:
        hsi: 高光譜影像 [C, H, W]
        r_band, g_band, b_band: R, G, B 通道索引
        ax: matplotlib 軸對象
        title: 圖的標題
        
    Returns:
        ax: matplotlib 軸對象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 選擇三個通道作為RGB
    rgb = np.zeros((hsi.shape[1], hsi.shape[2], 3))
    rgb[:, :, 0] = hsi[r_band]
    rgb[:, :, 1] = hsi[g_band]
    rgb[:, :, 2] = hsi[b_band]
    
    # 標準化到 [0, 1]
    for i in range(3):
        channel = rgb[:, :, i]
        p2, p98 = np.percentile(channel, (2, 98))
        rgb[:, :, i] = np.clip((channel - p2) / (p98 - p2), 0, 1)
    
    ax.imshow(rgb)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'RGB Composite (R:{r_band}, G:{g_band}, B:{b_band})')
    
    return ax


def visualize_segmentation_mask(mask, ax=None, title=None, alpha=0.7, cmap=None):
    """
    可視化分割掩膜
    
    Args:
        mask: 分割掩膜 [H, W]，整數表示類別
        ax: matplotlib 軸對象
        title: 圖的標題
        alpha: 透明度
        cmap: 色彩映射
        
    Returns:
        ax: matplotlib 軸對象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    if cmap is None:
        # 使用默認分割掩膜色彩映射
        colors = np.array([
            [0, 0, 0],           # 背景 (黑色)
            [1, 0, 0],           # 偽造區域 (紅色)
            [0, 1, 0],           # 類別2 (綠色)
            [0, 0, 1],           # 類別3 (藍色)
            [1, 1, 0],           # 類別4 (黃色)
            [1, 0, 1],           # 類別5 (洋紅色)
            [0, 1, 1],           # 類別6 (青色)
            [0.5, 0.5, 0.5]      # 類別7 (灰色)
        ])
        cmap = ListedColormap(colors)
    
    im = ax.imshow(mask, cmap=cmap, interpolation='nearest', alpha=alpha)
    
    # 添加圖例
    unique_values = np.unique(mask)
    if len(unique_values) <= 8:  # 只在類別較少時添加圖例
        patches = [plt.Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in unique_values]
        ax.legend(patches, [f'Class {i}' for i in unique_values], loc='upper right')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Segmentation Mask')
    
    return ax


def visualize_results(image, label, prediction, save_path=None, show=False):
    """
    可視化分割結果
    
    Args:
        image: 高光譜影像 [C, H, W]
        label: 真實標籤 [H, W]
        prediction: 預測結果 [H, W]
        save_path: 保存路徑
        show: 是否顯示圖像
        
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 顯示原始影像（假彩色）
    visualize_rgb_from_hsi(image, ax=axes[0], title='Original Image')
    
    # 顯示真實標籤
    visualize_segmentation_mask(label, ax=axes[1], title='Ground Truth')
    
    # 顯示預測結果
    visualize_segmentation_mask(prediction, ax=axes[2], title='Prediction')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_adversarial(original, adversarial, save_path=None):
    """
    可視化原始圖像、對抗樣本和它們之間的差異（放大50倍）
    Args:
        original: 原始圖像 (C, H, W)
        adversarial: 對抗樣本 (C, H, W)
        save_path: 保存路徑
    """
    # 轉換為 numpy 數組
    original = original.cpu().numpy() if torch.is_tensor(original) else original
    adversarial = adversarial.cpu().numpy() if torch.is_tensor(adversarial) else adversarial
    
    # 計算擾動
    perturbation = adversarial - original
    l2_norm = np.sqrt(np.sum(perturbation**2))
    
    # 確保數組是 3D 的
    if original.ndim == 2:
        original = original[np.newaxis, :, :]
    if adversarial.ndim == 2:
        adversarial = adversarial[np.newaxis, :, :]
    
    # 選擇三個波段進行可視化
    if original.shape[0] > 3:
        # 選擇具有最大方差的波段
        variances = np.var(original, axis=(1, 2))
        selected_bands = np.argsort(variances)[-3:]
        original = original[selected_bands]
        adversarial = adversarial[selected_bands]
        # 同樣選擇相應的擾動波段
        perturbation = perturbation[selected_bands]
    
    # 正規化每個通道，使用一致的範圍
    original_rgb = np.zeros((original.shape[1], original.shape[2], 3))
    adversarial_rgb = np.zeros((adversarial.shape[1], adversarial.shape[2], 3))
    
    for i in range(3):
        # 合併原始和對抗數據，計算統一的百分位數
        combined_data = np.concatenate([original[i].flatten(), adversarial[i].flatten()])
        p2 = np.percentile(combined_data, 2)
        p98 = np.percentile(combined_data, 98)
        
        if p98 - p2 > 1e-6:  # 避免除以接近零的值
            original_rgb[:, :, i] = np.clip((original[i] - p2) / (p98 - p2), 0, 1)
            adversarial_rgb[:, :, i] = np.clip((adversarial[i] - p2) / (p98 - p2), 0, 1)
        else:
            original_rgb[:, :, i] = np.clip(original[i], 0, 1)
            adversarial_rgb[:, :, i] = np.clip(adversarial[i], 0, 1)
    
    # 處理擾動可視化 - 放大50倍
    perturbation_vis = np.zeros((original.shape[1], original.shape[2], 3))
    # 放大差異50倍
    magnification = 50.0
    scaled_perturbation = perturbation * magnification
    
    # 將放大後的擾動轉換為RGB表示
    for i in range(3):
        # 正規化放大後的擾動到 [0, 1] 範圍
        # 使用絕對值便於可視化
        max_pert = np.max(np.abs(scaled_perturbation[i])) if np.max(np.abs(scaled_perturbation[i])) > 1e-6 else 1.0
        norm_pert = np.abs(scaled_perturbation[i]) / max_pert
        perturbation_vis[:, :, i] = np.clip(norm_pert, 0, 1)
    
    # 創建一個圖形，包含三個子圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 顯示原始圖像
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # 顯示對抗樣本
    axes[1].imshow(adversarial_rgb)
    axes[1].set_title("Adversarial Image", fontsize=14)
    axes[1].axis('off')
    
    # 顯示擾動（放大50倍）
    axes[2].imshow(perturbation_vis)
    axes[2].set_title("Difference × 50", fontsize=14)
    axes[2].axis('off')
    
    # 添加L2 norm信息
    fig.suptitle(f"L2 Norm: {l2_norm:.4f}", fontsize=12)
    
    plt.tight_layout()
    
    # 保存圖像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    return fig


def visualize_attack_comparison(original, adv_results, save_path=None, show=False):
    """
    可視化不同攻擊方法的對比
    
    Args:
        original: 原始高光譜影像 [C, H, W]
        adv_results: 字典，包含不同攻擊方法的結果
            {
                'method_name': {
                    'image': 對抗影像 [C, H, W],
                    'prediction': 對抗預測 [H, W]
                },
                ...
            }
        save_path: 保存路徑
        show: 是否顯示圖像
        
    Returns:
        None
    """
    methods = list(adv_results.keys())
    num_methods = len(methods)
    
    # 設置行數和列數
    cols = min(4, num_methods + 1)  # +1 是為了原始影像
    rows = (num_methods + cols) // cols
    if num_methods + 1 <= cols:
        rows = 2  # 至少2行
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1, 1) if cols == 1 else axes.reshape(1, -1)
    
    # 顯示原始影像
    visualize_rgb_from_hsi(original, ax=axes[0, 0], title='Original Image')
    
    # 在剩餘的位置顯示不同攻擊方法的結果
    idx = 1
    for method_name, result in adv_results.items():
        row, col = idx // cols, idx % cols
        if row < rows and col < cols:  # 確保索引在範圍內
            visualize_rgb_from_hsi(
                result['image'], 
                ax=axes[row, col], 
                title=f'{method_name}'
            )
        idx += 1
    
    # 隱藏空白子圖
    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= num_methods + 1:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_spectral_signature(hsi, mask=None, num_samples=5, save_path=None, show=False):
    """
    可視化高光譜影像的光譜特徵
    
    Args:
        hsi: 高光譜影像 [C, H, W]
        mask: 分割掩膜 [H, W]，用於選擇不同類別的像素
        num_samples: 每個類別選擇的樣本數
        save_path: 保存路徑
        show: 是否顯示圖像
        
    Returns:
        None
    """
    C, H, W = hsi.shape
    wavelengths = np.arange(C)  # 假設波長從0到C-1
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if mask is None:
        # 隨機選擇一些像素
        pixels = []
        for _ in range(num_samples):
            h, w = np.random.randint(0, H), np.random.randint(0, W)
            pixels.append(hsi[:, h, w])
        
        # 繪製光譜
        for i, pixel in enumerate(pixels):
            ax.plot(wavelengths, pixel, alpha=0.7, label=f'Pixel {i+1}')
        
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Reflectance')
        ax.set_title('Spectral Signatures of Random Pixels')
        
    else:
        # 獲取不同類別
        classes = np.unique(mask)
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        
        # 為每個類別選擇像素
        for i, cls in enumerate(classes):
            # 找到屬於該類別的像素位置
            positions = np.where(mask == cls)
            
            if len(positions[0]) == 0:
                continue
                
            # 隨機選擇一些位置
            indices = np.random.choice(len(positions[0]), min(num_samples, len(positions[0])), replace=False)
            
            # 繪製每個選定位置的光譜
            for j in indices:
                h, w = positions[0][j], positions[1][j]
                ax.plot(wavelengths, hsi[:, h, w], color=colors[i], alpha=0.5)
            
            # 繪製該類別的平均光譜
            pixels = [hsi[:, positions[0][j], positions[1][j]] for j in range(len(positions[0]))]
            if pixels:
                mean_spectrum = np.mean(pixels, axis=0)
                ax.plot(wavelengths, mean_spectrum, color=colors[i], linewidth=2, label=f'Class {cls}')
        
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Reflectance')
        ax.set_title('Spectral Signatures by Class')
    
    ax.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig) 