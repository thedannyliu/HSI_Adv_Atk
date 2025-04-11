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
    獲取當前時間的字串表示
    
    Returns:
        str: 當前時間的字串表示，格式為 YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_dataloader(batch_size, num_workers, root_dir=None, split='train', real_data_path=None):
    """
    獲取數據加載器
    
    Args:
        batch_size (int): 批次大小
        num_workers (int): 數據加載的工作進程數
        root_dir (str): 數據根目錄，如果為None，則使用項目根目錄
        split (str): 數據集分割，可為'train', 'val', 'test'
        real_data_path (str): 實際數據路徑，如果提供則優先使用此路徑
    
    Returns:
        DataLoader or tuple: 如果 split 為 'train'，則返回 (train_loader, val_loader)，否則只返回對應的加載器
    """
    from datasets.dataset import forgeryHSIDataset
    
    # 獲取項目根目錄
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 如果沒有指定根目錄，則使用項目根目錄
    if root_dir is None:
        if real_data_path:
            root_dir = real_data_path
            print(f"使用實際數據路徑: {root_dir}")
        else:
            root_dir = project_root
            print(f"未指定數據根目錄，使用項目根目錄: {root_dir}")
    
    # 獲取指定分割的加載器
    def get_loader(current_split):
        # 文件列表路徑，使用項目根目錄中的文件
        flist_path = os.path.join(project_root, f'{current_split}_all.txt')
        
        # 構建數據集
        dataset = forgeryHSIDataset(
            root=root_dir,
            flist=flist_path,
            split=current_split,
            target_type='mask',
            transform=None
        )
        
        # 構建數據加載器
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(current_split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        return loader
    
    # 如果請求訓練分割，則同時返回訓練和驗證加載器
    if split == 'train':
        train_loader = get_loader('train')
        val_loader = get_loader('val')
        return train_loader, val_loader
    else:
        # 否則返回請求的單個加載器
        return get_loader(split)

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