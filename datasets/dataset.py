import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image # Used for potential image loading if needed, though DFCN reads npy
import random as rn

# Helper function to normalize HSI data to RGB for visualization (like in DFCN's checkPerformance)
def normColor(R):
    """Normalize HSI cube to an RGB image for visualization."""
    # Example band selection for RGB visualization (adjust as needed)
    # These might not be the best bands for your specific AVIRIS data
    band_selection = [60, 27, 17] # Example: R, G, B bands indices (0-based)
    if R.shape[2] < max(band_selection) + 1:
        print(f"[WARN] normColor: Not enough bands ({R.shape[2]}) for selection {band_selection}. Using first 3 bands.")
        band_selection = [0, 1, 2]
        if R.shape[2] < 3:
             band_selection = [0, 0, 0] # Fallback for single channel
             
    R_rgb = R[:, :, band_selection].astype(np.float32)
    
    # Normalize each channel independently (common practice)
    for i in range(3):
        min_val = np.min(R_rgb[:, :, i])
        max_val = np.max(R_rgb[:, :, i])
        if max_val > min_val:
             R_rgb[:, :, i] = (R_rgb[:, :, i] - min_val) / (max_val - min_val)
        else:
             R_rgb[:, :, i] = 0 # Handle constant channel
             
    R_rgb = np.clip(R_rgb, 0, 1)
    R_rgb = (R_rgb * 255).astype(np.uint8)
    return R_rgb

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

    # 定義彩色映射表，用於 decode_target 方法 (與DFCN_Py/HRNetV2/dataset/dataset.py保持一致)
    # 類別：真實(0)為棕色，偽造(1)為紫色
    train_id_to_color = [(111, 74, 0), (81, 0, 81)]

    @classmethod
    def decode_target(cls, target):
        """
        將二進制標籤轉換為RGB彩色圖像，用於可視化。
        參數：
            target (np.ndarray): 形狀為(H, W)的標籤圖像，數值為0或1，分別代表真實和偽造。
        
        返回：
            np.ndarray: 形狀為(H, W, 3)的RGB彩色圖像，用於可視化。
        """
        target = target.astype(np.uint8)
        # 使用NumPy的廣播特性和色彩映射表實現快速轉換
        rgb = np.array(cls.train_id_to_color)[target]
        return rgb

    def __init__(self, root, flist, split='train', target_type='mask', transform=None, config='Origin'):
        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.config = config
        
        # 檢查flist是否為完整路徑
        if os.path.isabs(flist) and os.path.exists(flist):
            self.flist = flist
        elif os.path.exists(flist):
            # 如果是相對路徑但存在
            self.flist = flist
        else:
            # 使用以前的邏輯：相對於root/dataset_split_txt的路徑
            self.flist = os.path.join(root, 'dataset_split_txt', flist)
            
            # 如果仍然不存在，嘗試在項目根目錄查找
            if not os.path.exists(self.flist):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                project_flist = os.path.join(project_root, os.path.basename(flist))
                if os.path.exists(project_flist):
                    self.flist = project_flist
                else:
                    raise FileNotFoundError(f"找不到文件列表: {flist}, {self.flist}, {project_flist}")
        
        print(f"使用文件列表: {self.flist}")
        
        # 讀取文件列表和mask區域資訊
        self.img_paths = []
        self.mask_areas = []
        
        with open(self.flist, 'r') as f:
            for line in f:
                parts = line.strip().split(',', 1)  # 只分割第一个逗号
                
                if len(parts) >= 1:
                    img_path = parts[0]  # 例如 "config1/f090706t01p00r06_5_inpaint_result(1).npy"
                    
                    # 提取配置和文件名
                    if '/' in img_path:
                        config_dir, img_name = img_path.split('/', 1)
                    else:
                        config_dir = self.config
                        img_name = img_path
                    
                    # 将mask区域信息作为整行的剩余部分
                    mask_area = parts[1] if len(parts) > 1 else ""
                    
                    self.img_paths.append((config_dir, img_name))
                    self.mask_areas.append(mask_area)
        
        # 使用项目根目录的info_csv.csv
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        info_csv_path = os.path.join(project_root, 'info_csv.csv')
        try:
            self.info_df = pd.read_csv(info_csv_path)
            # 将 'sam' 列转换为字符串类型
            self.info_df['sam'] = self.info_df['sam'].astype(str)
            print(f"[INFO] 成功加载 info_csv.csv，共 {len(self.info_df)} 行")
        except Exception as e:
            print(f"[WARNING] 加载 info_csv.csv 出错: {str(e)}，将使用空的 DataFrame")
            self.info_df = pd.DataFrame(columns=['sam', 'mask_area'])
        
        print(f"[INFO] 载入 {split} 数据集，共 {len(self.img_paths)} 个样本")

    def __len__(self):
        return len(self.img_paths)

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
        # 获取图像路径和配置
        config_dir, img_name = self.img_paths[idx]
        mask_area = self.mask_areas[idx]
        
        # 獲取項目根目錄
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # 使用用戶提供的實際資料路徑
        real_data_root = '/ssd8/HyperForensics_Data/HyperForensics_Dataset'
        
        # 構建完整的圖像路徑
        if config_dir == 'Origin':
            # 嘗試多種可能的路徑，首先嘗試實際資料路徑
            possible_paths = [
                os.path.join(real_data_root, 'Origin', img_name),
                os.path.join(self.root, 'Origin', img_name),
                os.path.join(project_root, 'data', 'Origin', img_name),
                os.path.join(project_root, 'Origin', img_name)
            ]
        else:
            # 嘗試多種可能的路徑，首先嘗試實際資料路徑
            possible_paths = [
                os.path.join(real_data_root, 'ADMM_ADAM', config_dir, img_name),
                os.path.join(self.root, config_dir, img_name),
                os.path.join(self.root, 'ADMM_ADAM', config_dir, img_name),
                os.path.join(project_root, 'data', 'ADMM_ADAM', config_dir, img_name),
                os.path.join(project_root, 'data', config_dir, img_name),
                os.path.join(project_root, 'ADMM_ADAM', config_dir, img_name),
                os.path.join(project_root, config_dir, img_name)
            ]
        
        # 嘗試每個可能的路徑
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        # 檢查文件是否存在
        if img_path is None:
            raise FileNotFoundError(f"找不到文件，嘗試了以下路徑：{possible_paths}")

        # 读取图像 => shape = [256, 256, 172]
        try:
            # 使用 numpy 的 load 函数读取 .npy 文件
            if img_path.endswith('.npy'):
                image = np.load(img_path, allow_pickle=True)
            # 使用 fromfile 读取二进制文件
            else:
                try:
                    image = np.fromfile(img_path, dtype=np.float32).reshape(256, 256, 172)
                except:
                    # 尝试使用不同的数据类型
                    image = np.fromfile(img_path, dtype=np.float64).reshape(256, 256, 172)
                    image = image.astype(np.float32)
            
            image = np.ascontiguousarray(image)  # 确保内存连续
        except Exception as e:
            print(f"读取文件时发生错误 {img_path}: {str(e)}")
            raise
        
        if image.shape != (256, 256, 172):
            raise ValueError(f"{img_path} has shape {image.shape} != (256, 256, 172).")

        # 转置为 [Bands, H, W] => (172, 256, 256)
        image = image.transpose(2, 0, 1)

        # 创建标签 (H, W) = (256, 256)
        trg = np.zeros((256, 256), dtype=np.uint8)
        
        # 如果不是原始图片，则根据 mask_area 生成标签
        if config_dir != 'Origin' and mask_area:
            try:
                parts = mask_area.split(',')
                if len(parts) >= 2:
                    y_ranges = parts[0].strip()
                    x_ranges = parts[1].strip()
                    
                    # 处理多个y范围（用分号分隔）
                    for y_range in y_ranges.split(';'):
                        if ':' in y_range:
                            y_parts = y_range.split(':')
                            y1 = int(y_parts[0]) if y_parts[0].strip() else 0
                            y2 = int(y_parts[1]) if y_parts[1].strip() else 256
                        else:
                            if y_range.strip():
                                y1 = int(y_range.strip())
                                y2 = y1 + 1
                            else:
                                continue
                        
                        # 处理多个x范围（用分号分隔）
                        for x_range in x_ranges.split(';'):
                            if ':' in x_range:
                                x_parts = x_range.split(':')
                                x1 = int(x_parts[0]) if x_parts[0].strip() else 0
                                x2 = int(x_parts[1]) if x_parts[1].strip() else 256
                            else:
                                if x_range.strip():
                                    x1 = int(x_range.strip())
                                    x2 = x1 + 1
                                else:
                                    continue
                            
                            # 确保坐标在有效范围内
                            y1 = max(0, min(y1, 255))
                            y2 = max(1, min(y2, 256))
                            x1 = max(0, min(x1, 255))
                            x2 = max(1, min(x2, 256))
                            
                            trg[y1:y2, x1:x2] = 1  # 将伪造区域标记为 1
            except Exception as e:
                print(f"处理 mask_area 时发生错误 {mask_area}: {str(e)}")
                # 如果处理失败但有mask_area，将整个图像标记为伪造
                if mask_area:
                    trg = np.ones((256, 256), dtype=np.uint8)
                    print(f"将 {img_name} 全图标记为伪造（处理mask_area失败）")
        elif config_dir != 'Origin' and not mask_area:
            # 尝试从info_csv.csv获取mask_area
            mask_area_from_csv = self.get_mask_area(img_name)
            if mask_area_from_csv:
                try:
                    # 处理从CSV获取的mask_area（类似上面的逻辑）
                    parts = mask_area_from_csv.replace('"', '').split(',')
                    if len(parts) >= 2:
                        y_range = parts[0].strip()
                        x_range = parts[1].strip()
                        
                        # 处理 y 坐标
                        if ':' in y_range:
                            y_parts = y_range.split(':')
                            y1 = int(y_parts[0]) if y_parts[0].strip() else 0
                            y2 = int(y_parts[1]) if y_parts[1].strip() else 256
                        else:
                            y1, y2 = 0, 256
                        
                        # 处理 x 坐标
                        if ':' in x_range:
                            x_parts = x_range.split(':')
                            x1 = int(x_parts[0]) if x_parts[0].strip() else 0
                            x2 = int(x_parts[1]) if x_parts[1].strip() else 256
                        else:
                            x1, x2 = 0, 256
                        
                        # 确保坐标在有效范围内
                        y1 = max(0, min(y1, 255))
                        y2 = max(1, min(y2, 256))
                        x1 = max(0, min(x1, 255))
                        x2 = max(1, min(x2, 256))
                        
                        trg[y1:y2, x1:x2] = 1  # 将伪造区域标记为 1
                    print(f"使用从CSV获取的mask_area: {mask_area_from_csv}")
                except Exception as e:
                    print(f"处理CSV中的mask_area时发生错误: {str(e)}")
                    trg = np.ones((256, 256), dtype=np.uint8)
            else:
                # 如果找不到mask_area，将整个图像标记为伪造
                trg = np.ones((256, 256), dtype=np.uint8)
                print(f"找不到 {img_name} 的mask_area信息，将全图标记为伪造")

        # 如果有指定 transform，则应能处理 (image, trg)
        if self.transform:
            image, trg = self.transform(image, trg)

        # 转为 Tensor，确保使用 float32 类型
        image = torch.from_numpy(image.astype(np.float32))
        trg = torch.from_numpy(trg)
        
        # 为了与训练循环保持一致，添加一个band label（类似DFCN中的band regularization）
        # 这里简单用一个全零向量代替，实际应根据需求设定
        band_label = torch.zeros(172, dtype=torch.float32)
        
        return image, trg, band_label 