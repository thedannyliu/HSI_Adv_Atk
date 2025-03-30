import os
import numpy as np
import torch
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

        with open(flist, 'r') as f:
            lines = f.readlines()
        # 處理檔案名稱，移除引號和逗號，並確保移除所有空白字元
        self.data = []
        for line in lines:
            if line.strip():
                # 移除所有引號、逗號和空白字元
                clean_name = line.strip().replace('"', '').replace(',', '').strip()
                if clean_name:  # 確保清理後的名稱不為空
                    self.data.append(clean_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 取得影像路徑
        img_name = self.data[idx]
        
        # 根據配置類型決定資料路徑
        if self.config == 'Origin':
            img_path = os.path.join(self.root, 'Origin', img_name)
        else:
            img_path = os.path.join(self.root, 'ADMM_ADAM', self.config, img_name)

        # 檢查檔案是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到檔案：{img_path}")

        # 讀取影像 => shape = [256, 256, 172]
        try:
            # 使用 allow_pickle=True 並確保轉換為本機位元順序
            image = np.load(img_path, allow_pickle=True)
            image = np.ascontiguousarray(image)  # 確保記憶體連續
        except Exception as e:
            print(f"讀取檔案時發生錯誤 {img_path}: {str(e)}")
            raise
        
        if image.shape != (256, 256, 172):
            raise ValueError(f"{img_path} has shape {image.shape} != (256, 256, 172).")

        # 轉置為 [Bands, H, W] => (172, 256, 256)
        image = image.transpose(2, 0, 1)

        # 建立假標籤 (H, W) = (256, 256)
        trg = np.zeros((256, 256), dtype=np.uint8)
        
        # 如果不是原始圖片，則標記為偽造區域
        if self.config != 'Origin':
            # 這裡可以根據需要設定不同的標籤生成邏輯
            # 例如：根據config類型設定不同的標籤值
            trg = np.ones((256, 256), dtype=np.uint8)

        # 如果有指定 transform，則應能處理 (image, trg)
        if self.transform:
            image, trg = self.transform(image, trg)

        # 轉為 Tensor，確保使用 float32 型別
        image = torch.from_numpy(image.astype(np.float32))
        trg = torch.from_numpy(trg)

        return image, trg, img_name 