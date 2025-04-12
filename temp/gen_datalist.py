# Python script to generate train_all.txt and val_all.txt

import os
import csv
import random
import re

# --- Configuration ---
# !! 請確保這些路徑對於您的環境是正確的 !!
CSV_PATH = '/ssd1/dannyliu/HSI_Adv_Atk/HyperForensics++/info_csv.csv'
DATASET_ROOT = '/ssd8/HyperForensics_Data/HyperForensics_Dataset/ADMM_ADAM/'
OUTPUT_DIR = '/ssd1/dannyliu/HSI_Adv_Atk/HyperForensics++/config' # 修正輸出目錄
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train_all.txt')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val_all.txt')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test_all.txt')  # 新增測試集文件
CONFIG_DIRS = ['config0', 'config1', 'config2', 'config3', 'config4']
TRAIN_SPLIT_RATIO = 0.70  # 70% 用於訓練
VAL_SPLIT_RATIO = 0.15    # 15% 用於驗證
TEST_SPLIT_RATIO = 0.15   # 15% 用於測試
RANDOM_SEED = 42 # 用於可重現的劃分

# Config 目錄到波段範圍的映射 (根據您的描述)
CONFIG_TO_BANDS = {
    'config0': '0:172',        # 配置 1: 所有 172 個波段
    'config1': '50:90',        # 配置 2: 波段 50-90
    'config2': '100:130',      # 配置 3: 波段 100-130
    'config3': '140:172',      # 配置 4: 波段 140-172
    'config4': '10:20;40:50;80:90' # 配置 5: 多段
}

# --- Helper Function ---
def parse_mask_area(mask_str):
    """
    解析 CSV 中的 mask_area 字符串並轉換為 y1:y2;x1:x2 格式。
    特別處理如 ":25, 110:130, :" 這樣的格式。
    """
    if not isinstance(mask_str, str) or not mask_str.strip():
        return None
    
    # 清理字符串
    mask_str = mask_str.strip().strip('"')
    
    # 定義正則表達式來提取兩個範圍
    ranges = re.findall(r'(\d*:\d*)', mask_str)
    
    # 過濾掉空的範圍 (例如 ":")
    valid_ranges = [r for r in ranges if r and r != ':']
    
    if len(valid_ranges) >= 2:
        # 取前兩個有效範圍作為 y 和 x
        return f"{valid_ranges[0]};{valid_ranges[1]}"
    
    # 如果只找到一個範圍或沒有找到範圍
    print(f"[WARN] 無法從 '{mask_str}' 中提取兩個有效範圍")
    return None

# --- Main Logic ---
print("開始生成數據列表...")
random.seed(RANDOM_SEED)

# 1. Read CSV and create lookup
mask_lookup = {}
print(f"讀取CSV: {CSV_PATH}")
try:
    with open(CSV_PATH, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader) # Skip header
        for i, row in enumerate(reader):
            if len(row) > 4:
                filename = row[0].strip()
                mask_area_raw = row[4]
                
                # 使用改進的解析函數
                mask_area_parsed = parse_mask_area(mask_area_raw)
                
                if filename and mask_area_parsed:
                    mask_lookup[filename] = mask_area_parsed
                    # 減少輸出日誌，每50個輸出一次
                    if i % 50 == 0:
                        print(f"[INFO] 成功解析示例: {filename} -> {mask_area_parsed}")
except FileNotFoundError:
    print(f"[ERROR] CSV文件未找到: {CSV_PATH}")
    exit(1)
except Exception as e:
    print(f"[ERROR] 讀取或解析CSV失敗 {CSV_PATH}: {e}")
    exit(1)

print(f"從CSV中加載了 {len(mask_lookup)} 個有效條目。")

# 2. Iterate through config dirs, match, and format
all_samples = []
print(f"掃描數據目錄: {DATASET_ROOT}")

# 檢查DATASET_ROOT是否存在
if not os.path.isdir(DATASET_ROOT):
    print(f"[ERROR] 數據根目錄不存在: {DATASET_ROOT}")
    print(f"請檢查路徑是否正確，或者數據是否已下載。")
    exit(1)

for config_dir in CONFIG_DIRS:
    current_dir_path = os.path.join(DATASET_ROOT, config_dir)
    band_range = CONFIG_TO_BANDS.get(config_dir)

    if not band_range:
        print(f"[WARN] 未定義 {config_dir} 的波段範圍，跳過。")
        continue

    if not os.path.isdir(current_dir_path):
        print(f"[WARN] 目錄未找到: {current_dir_path}, 跳過。")
        continue

    print(f"處理目錄: {config_dir}")
    files_processed = 0
    files_missing = 0
    
    # 列出目錄中的所有文件，打印其中的一些作為樣本
    all_files = os.listdir(current_dir_path)
    sample_files = all_files[:5] if len(all_files) > 5 else all_files
    print(f"  目錄中的樣本文件: {sample_files}")
    
    for filename_npy in all_files:
        if filename_npy.endswith('.npy'):
            base_filename = os.path.splitext(filename_npy)[0]
            
            # 檢查文件名是否有前綴或後綴需要去除
            # 例如: 如果文件名為"image_f080107t01p00r08_38.npy"，但CSV中的鍵是"f080107t01p00r08_38"
            for csv_key in mask_lookup.keys():
                if csv_key in base_filename:
                    # 找到匹配的CSV鍵
                    mask_coords = mask_lookup.get(csv_key)
                    relative_path = os.path.join(config_dir, filename_npy).replace('\\', '/') # Use forward slashes
                    sample_line = f"{relative_path},{mask_coords},{band_range}"
                    all_samples.append(sample_line)
                    files_processed += 1
                    break
            else:
                # 如果直接匹配失敗，嘗試常規方式
                mask_coords = mask_lookup.get(base_filename)
                if mask_coords:
                    relative_path = os.path.join(config_dir, filename_npy).replace('\\', '/') # Use forward slashes
                    sample_line = f"{relative_path},{mask_coords},{band_range}"
                    all_samples.append(sample_line)
                    files_processed += 1
                else:
                    files_missing += 1
                    # 每50個缺失的文件顯示一次示例
                    if files_missing <= 3 or files_missing % 50 == 0:
                        print(f"  [INFO] 未找到對應的掩碼區域: {base_filename}")

    print(f"  找到 {files_processed} 個匹配樣本，{files_missing} 個文件沒有對應的掩碼區域。")

print(f"\n所有配置中共收集了 {len(all_samples)} 個樣本。")

# 3. Shuffle and Split
if not all_samples:
    print("[ERROR] 未收集到樣本。無法創建分割文件。")
    exit(1)

print("打亂樣本...")
random.shuffle(all_samples)

# 計算分割索引
train_end_idx = int(len(all_samples) * TRAIN_SPLIT_RATIO)
val_end_idx = train_end_idx + int(len(all_samples) * VAL_SPLIT_RATIO)

# 分割數據
train_samples = all_samples[:train_end_idx]
val_samples = all_samples[train_end_idx:val_end_idx]
test_samples = all_samples[val_end_idx:]

print(f"分割數據: {len(train_samples)} 個訓練樣本, {len(val_samples)} 個驗證樣本, {len(test_samples)} 個測試樣本。")

# 4. Write output files
try:
    print(f"寫入訓練數據到: {TRAIN_FILE}")
    with open(TRAIN_FILE, 'w') as f:
        for line in train_samples:
            f.write(line + '\n')  # 使用標準 \n 換行

    print(f"寫入驗證數據到: {VAL_FILE}")
    with open(VAL_FILE, 'w') as f:
        for line in val_samples:
            f.write(line + '\n')

    print(f"寫入測試數據到: {TEST_FILE}")
    with open(TEST_FILE, 'w') as f:
        for line in test_samples:
            f.write(line + '\n')

    print("\n生成完成!")
    print(f"  '{os.path.basename(TRAIN_FILE)}' 創建了 {len(train_samples)} 行。")
    print(f"  '{os.path.basename(VAL_FILE)}' 創建了 {len(val_samples)} 行。")
    print(f"  '{os.path.basename(TEST_FILE)}' 創建了 {len(test_samples)} 行。")

except IOError as e:
    print(f"[ERROR] 寫入輸出文件失敗: {e}")
    exit(1)
except Exception as e:
    print(f"[ERROR] 文件寫入過程中發生意外錯誤: {e}")
    exit(1) 