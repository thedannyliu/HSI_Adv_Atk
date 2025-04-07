import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接從相對位置導入
try:
    from data.dataset import HyperForensicsDataset
    from data.transforms import get_transforms
except ImportError:
    # 嘗試替代導入路徑
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(curr_dir)
    data_dir = os.path.join(project_root, 'data')
    sys.path.append(data_dir)
    
    # 嘗試列出可能的模塊位置
    print(f"嘗試在以下位置查找模塊:")
    print(f"當前目錄: {curr_dir}")
    print(f"項目根目錄: {project_root}")
    print(f"可能的數據目錄: {data_dir}")
    
    for root, dirs, files in os.walk(project_root):
        if 'dataset.py' in files:
            print(f"找到 dataset.py 在: {root}")
            sys.path.append(root)
            
            # 嘗試從找到的目錄導入
            parent_dir = os.path.basename(root)
            try:
                if parent_dir == 'data':
                    # 如果在 data 目錄下，嘗試直接導入
                    exec(f"from dataset import HyperForensicsDataset")
                    exec(f"from transforms import get_transforms")
                    print(f"成功導入模塊")
                else:
                    # 如果在其他目錄下，嘗試帶父目錄導入
                    exec(f"from {parent_dir}.dataset import HyperForensicsDataset")
                    exec(f"from {parent_dir}.transforms import get_transforms")
                    print(f"成功從 {parent_dir} 導入模塊")
                break
            except ImportError as e:
                print(f"無法從 {root} 導入: {e}")
                continue

def compute_mean_std(data_loader):
    """計算數據集的均值和標準差"""
    print("開始計算數據集統計信息...")
    mean_sum = torch.zeros(1)
    std_sum = torch.zeros(1)
    pixel_count = 0
    
    # 先計算全局均值
    for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="計算均值")):
        # images: [B, C, H, W]
        batch_size, channels, height, width = images.shape
        images_flat = images.view(batch_size, channels, -1)
        
        mean_sum += images_flat.mean(dim=2).sum(dim=0).sum(dim=0)
        pixel_count += batch_size
        
    global_mean = mean_sum / pixel_count
    
    # 再計算全局標準差
    variance_sum = torch.zeros(1)
    for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="計算標準差")):
        batch_size, channels, height, width = images.shape
        images_flat = images.view(batch_size, channels, -1)
        
        # 計算每個樣本的方差
        variance_sum += ((images_flat - global_mean.view(1, 1, 1)) ** 2).mean(dim=2).sum(dim=0).sum(dim=0)
        
    global_std = torch.sqrt(variance_sum / pixel_count)
    
    return global_mean.item(), global_std.item()

def compute_channel_mean_std(data_loader):
    """計算每個通道的均值和標準差"""
    print("開始計算每個通道的統計信息...")
    num_channels = None
    channel_sum = None
    channel_sum_squared = None
    pixel_count = 0
    
    # 第一步：計算總和和平方和
    for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc="計算通道統計")):
        # 獲取通道數
        if channel_sum is None:
            _, num_channels, _, _ = images.shape
            channel_sum = torch.zeros(num_channels)
            channel_sum_squared = torch.zeros(num_channels)
        
        # 計算每個通道的和
        batch_size, _, height, width = images.shape
        num_pixels = batch_size * height * width
        
        # 對每個通道計算和
        for c in range(num_channels):
            channel_data = images[:, c, :, :]
            channel_sum[c] += torch.sum(channel_data).item()
            channel_sum_squared[c] += torch.sum(channel_data ** 2).item()
        
        pixel_count += num_pixels
    
    # 計算均值和標準差
    channel_mean = channel_sum / pixel_count
    channel_std = torch.sqrt(channel_sum_squared / pixel_count - channel_mean ** 2)
    
    # 計算全局統計
    global_mean = torch.mean(channel_mean).item()
    global_std = torch.mean(channel_std).item()
    
    print(f"全局均值: {global_mean:.6f}")
    print(f"全局標準差: {global_std:.6f}")
    
    # 檢查異常值
    min_mean = torch.min(channel_mean).item()
    max_mean = torch.max(channel_mean).item()
    min_std = torch.min(channel_std).item()
    max_std = torch.max(channel_std).item()
    
    print(f"通道均值範圍: {min_mean:.6f} - {max_mean:.6f}")
    print(f"通道標準差範圍: {min_std:.6f} - {max_std:.6f}")
    
    return global_mean, global_std, channel_mean.numpy(), channel_std.numpy()

def update_config(config_path, mean, std):
    """更新配置文件中的均值和標準差"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['DATASET']['MEAN'] = mean
    config['DATASET']['STD'] = std
    
    # 保存更新後的配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已更新配置文件 {config_path} 中的MEAN={mean:.6f}和STD={std:.6f}")

def main():
    parser = argparse.ArgumentParser(description='計算數據集的均值和標準差')
    parser.add_argument('--config_path', type=str, default='./config/attack_config.yaml', help='配置文件路徑')
    parser.add_argument('--update_config', action='store_true', help='是否更新配置文件')
    args = parser.parse_args()
    
    # 加載配置
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 準備數據集
    train_transform, _ = get_transforms()
    dataset = HyperForensicsDataset(
        root=cfg['DATASET']['ROOT'],
        data_list=cfg['DATASET']['Train_data'],
        transform=train_transform,
        num_classes=cfg['DATASET']['NUM_CLASSES']
    )
    
    # 創建數據加載器
    data_loader = DataLoader(
        dataset,
        batch_size=cfg['ATTACK']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['ATTACK']['NUM_WORKERS'],
        pin_memory=True
    )
    
    # 計算統計信息
    global_mean, global_std, channel_mean, channel_std = compute_channel_mean_std(data_loader)
    
    # 保存統計信息
    stats_dir = os.path.join(os.path.dirname(args.config_path), 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    np.save(os.path.join(stats_dir, 'channel_mean.npy'), channel_mean)
    np.save(os.path.join(stats_dir, 'channel_std.npy'), channel_std)
    
    # 更新配置
    if args.update_config:
        update_config(args.config_path, global_mean, global_std)
    else:
        print("統計信息已計算完成。如需更新配置文件，請添加 --update_config 參數。")

if __name__ == '__main__':
    main() 