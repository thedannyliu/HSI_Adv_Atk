import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import argparse
import yaml
import glob

def load_file(filepath):
    """加載高光譜文件"""
    if filepath.endswith('.npy'):
        return np.load(filepath)
    else:
        print(f"不支持的文件格式: {filepath}")
        return None

def compute_stats_from_files(file_list, max_files=None):
    """從文件列表計算均值和標準差"""
    print("開始處理文件...")
    
    # 限制處理文件數量
    if max_files and len(file_list) > max_files:
        print(f"限制處理文件數為 {max_files}，總文件數: {len(file_list)}")
        file_list = file_list[:max_files]
    
    # 初始化變量
    sum_values = 0
    sum_squares = 0
    pixel_count = 0
    
    # 處理每個文件
    for file_path in tqdm(file_list, desc="處理文件"):
        try:
            # 加載數據
            data = load_file(file_path)
            if data is None:
                continue
                
            # 轉為浮點數
            data = data.astype(np.float32)
            
            # 累加值
            sum_values += np.sum(data)
            sum_squares += np.sum(data ** 2)
            pixel_count += data.size
            
        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            continue
    
    # 計算均值和標準差
    mean = sum_values / pixel_count
    variance = (sum_squares / pixel_count) - (mean ** 2)
    std = np.sqrt(variance)
    
    print(f"均值: {mean:.6f}")
    print(f"標準差: {std:.6f}")
    
    return mean, std

def compute_channel_stats_from_files(file_list, max_files=None):
    """從文件列表計算每個通道的均值和標準差"""
    print("開始處理文件計算通道統計...")
    
    # 限制處理文件數量
    if max_files and len(file_list) > max_files:
        print(f"限制處理文件數為 {max_files}，總文件數: {len(file_list)}")
        file_list = file_list[:max_files]
    
    # 初始化變量
    num_channels = None
    channel_sums = None
    channel_squares = None
    pixel_counts = None
    
    # 處理每個文件
    for file_path in tqdm(file_list, desc="處理文件"):
        try:
            # 加載數據
            data = load_file(file_path)
            if data is None:
                continue
                
            # 轉為浮點數
            data = data.astype(np.float32)
            
            # 確定通道數量（假設數據格式為 [C, H, W]）
            if len(data.shape) == 3:
                curr_channels = data.shape[0]
            else:
                print(f"警告: 文件 {file_path} 的維度不是3D，跳過")
                continue
                
            # 初始化數組
            if num_channels is None:
                num_channels = curr_channels
                channel_sums = np.zeros(num_channels, dtype=np.float64)
                channel_squares = np.zeros(num_channels, dtype=np.float64)
                pixel_counts = np.zeros(num_channels, dtype=np.int64)
            elif curr_channels != num_channels:
                print(f"警告: 文件 {file_path} 的通道數 ({curr_channels}) 與預期 ({num_channels}) 不符，跳過")
                continue
            
            # 計算每個通道的統計
            for c in range(num_channels):
                channel_data = data[c]
                channel_sums[c] += np.sum(channel_data)
                channel_squares[c] += np.sum(channel_data ** 2)
                pixel_counts[c] += channel_data.size
                
        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            continue
    
    # 計算均值和標準差
    channel_means = channel_sums / pixel_counts
    channel_variances = (channel_squares / pixel_counts) - (channel_means ** 2)
    channel_stds = np.sqrt(channel_variances)
    
    # 計算全局均值和標準差
    global_mean = np.mean(channel_means)
    global_std = np.mean(channel_stds)
    
    print(f"全局均值: {global_mean:.6f}")
    print(f"全局標準差: {global_std:.6f}")
    
    # 檢查異常值
    min_mean = np.min(channel_means)
    max_mean = np.max(channel_means)
    min_std = np.min(channel_stds)
    max_std = np.max(channel_stds)
    
    print(f"通道均值範圍: {min_mean:.6f} - {max_mean:.6f}")
    print(f"通道標準差範圍: {min_std:.6f} - {max_std:.6f}")
    
    return global_mean, global_std, channel_means, channel_stds

def update_config(config_path, mean, std):
    """更新配置文件中的均值和標準差"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['DATASET']['MEAN'] = float(mean)
    config['DATASET']['STD'] = float(std)
    
    # 保存更新後的配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已更新配置文件 {config_path} 中的MEAN={mean:.6f}和STD={std:.6f}")

def main():
    parser = argparse.ArgumentParser(description='計算數據集的均值和標準差')
    parser.add_argument('--config_path', type=str, default='./config/attack_config.yaml', help='配置文件路徑')
    parser.add_argument('--data_dir', type=str, help='數據目錄路徑')
    parser.add_argument('--data_pattern', type=str, default='*.npy', help='數據文件匹配模式')
    parser.add_argument('--max_files', type=int, default=None, help='最多處理的文件數量')
    parser.add_argument('--update_config', action='store_true', help='是否更新配置文件')
    args = parser.parse_args()
    
    # 加載配置
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 確定數據目錄
    data_dir = args.data_dir if args.data_dir else cfg['DATASET']['ROOT']
    
    # 查找所有匹配的文件
    file_pattern = os.path.join(data_dir, '**', args.data_pattern)
    file_list = glob.glob(file_pattern, recursive=True)
    
    if not file_list:
        print(f"錯誤: 在 {data_dir} 中沒有找到匹配 {args.data_pattern} 的文件")
        return
    
    print(f"找到 {len(file_list)} 個文件")
    
    # 計算統計信息
    global_mean, global_std, channel_means, channel_stds = compute_channel_stats_from_files(
        file_list, 
        max_files=args.max_files
    )
    
    # 保存統計信息
    stats_dir = os.path.join(os.path.dirname(args.config_path), 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    np.save(os.path.join(stats_dir, 'channel_mean.npy'), channel_means)
    np.save(os.path.join(stats_dir, 'channel_std.npy'), channel_stds)
    
    # 更新配置
    if args.update_config:
        update_config(args.config_path, global_mean, global_std)
    else:
        print("統計信息已計算完成。如需更新配置文件，請添加 --update_config 參數。")

if __name__ == '__main__':
    main() 