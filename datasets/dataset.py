import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class forgeryHSIDataset(Dataset):
    """
    高光譜影像偽造檢測數據集
    不做任何裁切、padding。影像維持 (256, 256, 172)，
    讀取後 transpose => (172, 256, 256)。
    
    Args:
        root (str): 數據根目錄
        flist (str): 文件列表路徑，每行一個文件名
        split (str): 數據集分割，可為'train', 'val', 'test'
        target_type (str): 目標類型，默認為'mask'
        transform: 數據轉換操作
        config (str): 配置類型，可為'Origin', 'config1', 'config2'等
    """

    def __init__(self, root, flist, split='train', target_type='mask', transform=None, config='Origin'):
        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.config = config
        
        # 組合完整的文件列表路徑
        self.flist = os.path.join(root, 'dataset_split_txt', flist)
        
        # 讀取文件列表
        with open(self.flist, 'r') as f:
            self.imgs = [line.strip() for line in f.readlines()]
        
        # 清理文件名
        self.imgs = [img.split('/')[-1] for img in self.imgs]
        
        # 讀取 info_csv，確保 'sam' 列為字符串類型
        self.info_df = pd.read_csv(os.path.join(root, 'info_csv.csv'))
        # 將 'sam' 列轉換為字符串類型
        self.info_df['sam'] = self.info_df['sam'].astype(str)
        
        print(f"[INFO] 載入 {config} 數據集，共 {len(self.imgs)} 個樣本")

    def __len__(self):
        return len(self.imgs)

    def get_mask_area(self, img_name):
        """從 info_csv 中獲取對應的 mask_area"""
        try:
            # 移除可能的副檔名
            img_name = os.path.splitext(img_name)[0]
            
            # 提取基本檔名（移除 _inpaint_result(1) 等後綴）
            base_name = img_name
            if "_inpaint_result" in img_name:
                base_name = img_name.split("_inpaint_result")[0]
            
            # 在 CSV 中尋找匹配（使用第一列 - CSV 的索引而不是 'sam' 欄位）
            # 檢查是否有 'Unnamed: 0' 欄位 (pandas 默認給第一列無名稱的列命名)
            if 'Unnamed: 0' in self.info_df.columns:
                row = self.info_df[self.info_df['Unnamed: 0'] == base_name]
                if not row.empty:
                    mask_area = row['mask_area'].iloc[0]
                    # print(f"找到 {base_name} 的 mask_area (使用第一列): {mask_area}")
                    return mask_area
            
            # 嘗試使用 CSV 的索引進行匹配（如果第一列被設為索引）
            try:
                if base_name in self.info_df.index:
                    mask_area = self.info_df.loc[base_name, 'mask_area']
                    # print(f"找到 {base_name} 的 mask_area (使用索引): {mask_area}")
                    return mask_area
            except:
                pass
            
            # 後備方案：仍然嘗試 'sam' 欄位（如果前面的方法失敗）
            row = self.info_df[self.info_df['sam'] == base_name]
            if not row.empty:
                mask_area = row['mask_area'].iloc[0]
                # print(f"找到 {base_name} 的 mask_area (使用 'sam' 欄位): {mask_area}")
                return mask_area
            
            # 只在失敗時打印一條簡短的警告
            # print(f"CSV 檔案前5行: {self.info_df.head()}")
            # print(f"CSV 檔案欄位: {self.info_df.columns.tolist()}")
            
            # 嘗試搜索所有欄位中是否含有該文件名
            for col in self.info_df.columns:
                try:
                    if self.info_df[col].astype(str).str.contains(base_name).any():
                        matches = self.info_df[self.info_df[col].astype(str).str.contains(base_name)]
                        if not matches.empty:
                            mask_area = matches['mask_area'].iloc[0]
                            matching_value = matches[col].iloc[0]
                            # print(f"在欄位 '{col}' 找到部分匹配 {base_name} -> {matching_value} 的 mask_area: {mask_area}")
                            return mask_area
                except:
                    continue
            
            print(f"警告: 找不到 {img_name} 的 mask_area")
            return None
        except Exception as e:
            print(f"在 get_mask_area 中發生未處理的錯誤: {str(e)}")
            return None

    def __getitem__(self, idx):
        # 取得影像路徑
        img_name = self.imgs[idx]
        
        # 根據配置類型決定資料路徑
        if self.config == 'Origin':
            img_path = os.path.join(self.root, 'Origin', img_name)
        elif self.config == 'config1':
            img_path = os.path.join(self.root, 'ADMM_ADAM', 'config1', img_name)
        else:
            img_path = os.path.join(self.root, 'ADMM_ADAM', self.config, img_name)

        # 檢查檔案是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到檔案：{img_path}")

        # 讀取影像 => shape = [256, 256, 172]
        try:
            # 使用 numpy 的 load 函數讀取 .npy 文件
            if img_path.endswith('.npy'):
                image = np.load(img_path, allow_pickle=True)
            # 使用 fromfile 讀取二進制文件
            else:
                try:
                    image = np.fromfile(img_path, dtype=np.float32).reshape(256, 256, 172)
                except:
                    # 嘗試使用不同的數據類型
                    image = np.fromfile(img_path, dtype=np.float64).reshape(256, 256, 172)
                    image = image.astype(np.float32)
            
            image = np.ascontiguousarray(image)  # 確保記憶體連續
        except Exception as e:
            print(f"讀取檔案時發生錯誤 {img_path}: {str(e)}")
            raise
        
        if image.shape != (256, 256, 172):
            raise ValueError(f"{img_path} has shape {image.shape} != (256, 256, 172).")

        # 轉置為 [Bands, H, W] => (172, 256, 256)
        image = image.transpose(2, 0, 1)

        # 建立標籤 (H, W) = (256, 256)
        trg = np.zeros((256, 256), dtype=np.uint8)
        
        # 如果不是原始圖片，則根據 mask_area 生成標籤
        if self.config != 'Origin':
            mask_area = self.get_mask_area(img_name)
            if mask_area:
                try:
                    parts = mask_area.replace('"', '').split(',')
                    if len(parts) >= 2:
                        y_range = parts[0].strip()
                        x_range = parts[1].strip()
                        
                        # 處理 y 座標
                        if ':' in y_range:
                            y_parts = y_range.split(':')
                            y1 = int(y_parts[0]) if y_parts[0].strip() else 0
                            y2 = int(y_parts[1]) if y_parts[1].strip() else 256
                        else:
                            y1, y2 = 0, 256
                        
                        # 處理 x 座標
                        if ':' in x_range:
                            x_parts = x_range.split(':')
                            x1 = int(x_parts[0]) if x_parts[0].strip() else 0
                            x2 = int(x_parts[1]) if x_parts[1].strip() else 256
                        else:
                            x1, x2 = 0, 256
                        
                        # 確保座標在有效範圍內
                        y1 = max(0, min(y1, 255))
                        y2 = max(1, min(y2, 256))
                        x1 = max(0, min(x1, 255))
                        x2 = max(1, min(x2, 256))
                        
                        trg[y1:y2, x1:x2] = 1  # 將偽造區域標記為 1
                        
                        # 只在 debug 模式下或每 100 次打印一次標籤統計
                        if idx % 100 == 0:
                            nonzero_count = np.count_nonzero(trg)
                            total_pixels = trg.size
                            print(f"樣本 {idx}: 圖像 {img_name} 的標籤統計: 偽造像素 {nonzero_count}/{total_pixels} ({nonzero_count/total_pixels*100:.2f}%)")
                except Exception as e:
                    print(f"處理 mask_area 時發生錯誤 {mask_area}: {str(e)}")
                    # 如果處理失敗，使用全圖標籤
                    trg = np.ones((256, 256), dtype=np.uint8)
                    print(f"將 {img_name} 全圖標記為偽造")
            else:
                # 如果找不到 mask_area，將整個圖像標記為偽造
                print(f"找不到 {img_name} 的 mask_area，將全圖標記為偽造")
                trg = np.ones((256, 256), dtype=np.uint8)

        # 如果有指定 transform，則應能處理 (image, trg)
        if self.transform:
            image, trg = self.transform(image, trg)

        # 轉為 Tensor，確保使用 float32 型別
        image = torch.from_numpy(image.astype(np.float32))
        trg = torch.from_numpy(trg)

        return image, trg, img_name 