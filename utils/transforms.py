import numpy as np
import torch
import random
import cv2
import torchvision.transforms.functional as F
from typing import Tuple
from PIL import Image

class DummyTransform:
    """
    空轉換類，不做任何操作
    """
    def __call__(self, image, mask=None):
        if mask is None:
            return image
        return image, mask

class PairCompose:
    """
    組合多個轉換操作
    
    Args:
        transforms: 轉換操作列表
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '     {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomCrop:
    """
    隨機裁剪
    
    Args:
        size (tuple): 裁剪大小 (height, width)
        smart_crop (bool): 是否優先包含偽造區域
    """
    def __init__(self, size, smart_crop=True):
        self.size = size
        self.smart_crop = smart_crop

    def __call__(self, image, mask):
        height, width = self.size
        if len(mask.shape) > 2:
            h, w = mask.shape[0], mask.shape[1]
        else:
            h, w = mask.shape
        
        # 如果使用智能裁剪且有偽造區域
        if self.smart_crop and np.any(mask == 1):
            # 找到偽造區域範圍
            if len(mask.shape) > 2:
                # 處理多通道mask
                mask_binary = np.max(mask, axis=2) > 0
                y_indices, x_indices = np.where(mask_binary)
            else:
                y_indices, x_indices = np.where(mask == 1)
                
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                
                # 計算有效的起始位置範圍，確保裁剪區域包含偽造部分
                max_start_y = min(y_min, h - height)
                min_start_y = max(0, y_max - height)
                max_start_x = min(x_min, w - width)
                min_start_x = max(0, x_max - width)
                
                # 如果範圍有效，則在範圍內隨機選擇起始位置
                if max_start_y >= min_start_y and max_start_x >= min_start_x:
                    start_h = random.randint(min_start_y, max_start_y) if min_start_y < max_start_y else min_start_y
                    start_w = random.randint(min_start_x, max_start_x) if min_start_x < max_start_x else min_start_x
                    
                    # 裁剪
                    if len(image.shape) == 3:  # CHW格式
                        image = image[:, start_h:start_h+height, start_w:start_w+width]
                    else:  # HWC格式
                        image = image[start_h:start_h+height, start_w:start_w+width, :]
                    
                    if len(mask.shape) > 2:
                        mask = mask[start_h:start_h+height, start_w:start_w+width, :]
                    else:
                        mask = mask[start_h:start_h+height, start_w:start_w+width]
                    
                    return image, mask
        
        # 如果沒有偽造區域或智能裁剪無法進行，則隨機裁剪
        max_h = h - height
        max_w = w - width
        
        if max_h < 0 or max_w < 0:
            return image, mask  # 如果裁剪尺寸大於原圖，則不裁剪
        
        # 隨機選擇起始點
        start_h = random.randint(0, max_h)
        start_w = random.randint(0, max_w)
        
        # 裁剪
        if len(image.shape) == 3 and image.shape[0] <= 3:  # CHW格式
            image = image[:, start_h:start_h+height, start_w:start_w+width]
        else:  # HWC格式或通道數較多的情況
            if len(image.shape) == 3 and image.shape[0] > 3:  # 多通道的CHW格式
                image = image[:, start_h:start_h+height, start_w:start_w+width]
            else:
                image = image[start_h:start_h+height, start_w:start_w+width]
        
        if len(mask.shape) > 2:
            mask = mask[start_h:start_h+height, start_w:start_w+width, :]
        else:
            mask = mask[start_h:start_h+height, start_w:start_w+width]
            
        return image, mask

class RandomFlip:
    """
    隨機翻轉
    
    Args:
        p_h (float): 水平翻轉概率
        p_v (float): 垂直翻轉概率
    """
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, image, mask):
        if random.random() < self.p_h:
            # 水平翻轉
            if len(image.shape) == 3 and image.shape[0] <= 3:  # CHW格式
                image = np.flip(image, axis=2).copy()
            else:  # HWC格式或多通道CHW
                if len(image.shape) == 3 and image.shape[0] > 3:  # 多通道的CHW格式
                    image = np.flip(image, axis=2).copy()
                else:
                    image = np.flip(image, axis=1).copy()
                    
            if len(mask.shape) > 2:
                mask = np.flip(mask, axis=1).copy()
            else:
                mask = np.flip(mask, axis=1).copy()
        
        if random.random() < self.p_v:
            # 垂直翻轉
            if len(image.shape) == 3 and image.shape[0] <= 3:  # CHW格式
                image = np.flip(image, axis=1).copy()
            else:  # HWC格式或多通道CHW
                if len(image.shape) == 3 and image.shape[0] > 3:  # 多通道的CHW格式
                    image = np.flip(image, axis=1).copy()
                else:
                    image = np.flip(image, axis=0).copy()
                    
            if len(mask.shape) > 2:
                mask = np.flip(mask, axis=0).copy()
            else:
                mask = np.flip(mask, axis=0).copy()
            
        return image, mask

class RandomRotation:
    """
    隨機旋轉
    
    Args:
        degrees (tuple): 旋轉角度範圍 (min, max)
    """
    def __init__(self, degrees=(0, 90)):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # 判斷圖像格式
        if len(image.shape) == 3 and image.shape[0] > 3:  # CHW格式，多通道
            channels, height, width = image.shape
            
            # 計算旋轉矩陣
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 對每個通道進行旋轉
            rotated_image = np.zeros_like(image)
            for c in range(channels):
                rotated_image[c] = cv2.warpAffine(image[c], matrix, (width, height))
        else:  # HWC格式或RGB圖像
            if len(image.shape) == 3 and image.shape[0] <= 3:  # CHW格式，RGB
                image = np.transpose(image, (1, 2, 0))  # 轉為HWC
                
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, matrix, (width, height))
            
            if len(image.shape) == 3 and image.shape[0] <= 3:  # 轉換回CHW
                rotated_image = np.transpose(rotated_image, (2, 0, 1))
        
        # 旋轉掩碼
        if len(mask.shape) > 2:
            height, width = mask.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_mask = cv2.warpAffine(mask, matrix, (width, height))
        else:
            height, width = mask.shape
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_mask = cv2.warpAffine(mask, matrix, (width, height))
        
        return rotated_image, rotated_mask

class PairToTensor:
    """
    將numpy數組轉換為PyTorch張量
    
    Args:
        normalize (bool): 是否將圖像標準化到[0,1]範圍
    """
    def __init__(self, normalize=True):
        self.normalize = normalize
        
    def __call__(self, image, mask):
        # 確保image是float32類型
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.astype(np.float32))
            
            # 如果需要標準化且不是張量
            if self.normalize and not isinstance(image, torch.Tensor):
                image = image / 255.0
        
        # 確保mask是long類型
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.astype(np.int64))
            
        return image, mask

class PairNormalize:
    """
    標準化圖像
    
    Args:
        mean (list): 均值
        std (list): 標準差
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, mask):
        # 確保image是張量
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.astype(np.float32))
            
        # 應用標準化
        if len(image.shape) == 3 and image.shape[0] == 3:  # RGB圖像，CHW格式
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        elif len(image.shape) == 3 and image.shape[0] > 3:  # 高光譜圖像，CHW格式
            # 對於高光譜數據，可以使用不同的標準化策略
            image = (image - torch.mean(image)) / (torch.std(image) + 1e-6)
            
        return image, mask

class PairRandomScale:
    """
    隨機縮放
    
    Args:
        scale_range (tuple): 縮放範圍
    """
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
        
    def __call__(self, image, mask):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 判斷圖像格式
        if len(image.shape) == 3 and image.shape[0] > 3:  # CHW格式，多通道
            channels, height, width = image.shape
            new_height, new_width = int(height * scale), int(width * scale)
            
            # 創建新圖像
            scaled_image = np.zeros((channels, new_height, new_width), dtype=image.dtype)
            
            # 對每個通道進行縮放
            for c in range(channels):
                scaled_image[c] = cv2.resize(image[c], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:  # HWC格式或RGB圖像
            if len(image.shape) == 3 and image.shape[0] <= 3:  # CHW格式，RGB
                image = np.transpose(image, (1, 2, 0))  # 轉為HWC
                
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            if len(image.shape) == 3 and image.shape[0] <= 3:  # 轉換回CHW
                scaled_image = np.transpose(scaled_image, (2, 0, 1))
        
        # 縮放掩碼
        if len(mask.shape) > 2:
            scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
        return scaled_image, scaled_mask 