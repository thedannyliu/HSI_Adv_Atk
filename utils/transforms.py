import numpy as np
import torch
import random
import cv2

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

class RandomCrop:
    """
    隨機裁剪
    
    Args:
        size (tuple): 裁剪大小 (height, width)
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        height, width = self.size
        h, w = mask.shape
        
        # 計算可能的起始點
        max_h = h - height
        max_w = w - width
        
        if max_h < 0 or max_w < 0:
            return image, mask  # 如果裁剪尺寸大於原圖，則不裁剪
        
        # 隨機選擇起始點
        start_h = random.randint(0, max_h)
        start_w = random.randint(0, max_w)
        
        # 裁剪
        image = image[:, start_h:start_h+height, start_w:start_w+width]
        mask = mask[start_h:start_h+height, start_w:start_w+width]
        
        return image, mask

class RandomFlip:
    """
    隨機翻轉
    
    Args:
        p (float): 翻轉概率
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            # 水平翻轉
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        if random.random() < self.p:
            # 垂直翻轉
            image = np.flip(image, axis=1).copy()
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
        
        # 取得圖像尺寸
        channels, height, width = image.shape
        
        # 計算旋轉矩陣
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 對每個通道進行旋轉
        rotated_image = np.zeros_like(image)
        for c in range(channels):
            rotated_image[c] = cv2.warpAffine(image[c], matrix, (width, height))
        
        # 旋轉掩碼
        rotated_mask = cv2.warpAffine(mask, matrix, (width, height))
        
        return rotated_image, rotated_mask 