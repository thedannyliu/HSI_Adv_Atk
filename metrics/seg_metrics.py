import numpy as np
import torch
import torch.nn.functional as F


class SegMetrics:
    """
    分割評估指標計算類
    
    計算包括:
    - 像素準確率 (Pixel Accuracy)
    - 平均IoU (Mean IoU)
    - 類別F1分數 (Class F1 Score)
    - 平均F1分數 (Mean F1 Score)
    """
    
    def __init__(self, n_classes, device=None):
        """
        初始化
        
        Args:
            n_classes: 類別數量
            device: 計算設備
        """
        self.n_classes = n_classes
        self.device = device if device is not None else torch.device('cpu')
        self.confusion_matrix = torch.zeros((n_classes, n_classes), device=self.device)
        self.num_samples = 0
        
    def _fast_hist(self, label_true, label_pred):
        """
        快速計算混淆矩陣
        
        Args:
            label_true: 真實標籤, shape=[B, H, W], dtype=long
            label_pred: 預測標籤, shape=[B, H, W], dtype=long
            
        Returns:
            torch.Tensor: 更新的混淆矩陣
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = torch.bincount(
            self.n_classes * label_true[mask] + label_pred[mask],
            minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        return hist
    
    def update(self, label_true, label_pred):
        """
        更新混淆矩陣
        
        Args:
            label_true: 真實標籤, shape=[B, H, W], dtype=long
            label_pred: 預測標籤, shape=[B, H, W], dtype=long
        """
        for lt, lp in zip(label_true, label_pred):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.num_samples += label_true.size(0)
    
    def pixel_accuracy(self):
        """
        計算像素準確率
        
        Returns:
            float: 像素準確率
        """
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc.item()
    
    def class_accuracy(self):
        """
        計算各類別的準確率
        
        Returns:
            numpy.ndarray: 各類別的準確率
        """
        acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return acc.cpu().numpy()
    
    def mean_accuracy(self):
        """
        計算平均準確率
        
        Returns:
            float: 平均準確率
        """
        acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1)
        return acc.mean().item()
    
    def mean_iou(self):
        """
        計算平均IoU
        
        Returns:
            float: 平均IoU
        """
        iou = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + 
            self.confusion_matrix.sum(dim=0) - 
            torch.diag(self.confusion_matrix)
        )
        return iou.mean().item()
    
    def class_iou(self):
        """
        計算各類別的IoU
        
        Returns:
            numpy.ndarray: 各類別的IoU
        """
        iou = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + 
            self.confusion_matrix.sum(dim=0) - 
            torch.diag(self.confusion_matrix)
        )
        return iou.cpu().numpy()
    
    def f1_score(self):
        """
        計算各類別的F1分數
        
        Returns:
            numpy.ndarray: 各類別的F1分數
        """
        precision = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=0).clamp(min=1e-6)
        recall = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(dim=1).clamp(min=1e-6)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
        return f1.cpu().numpy()
    
    def mean_f1(self):
        """
        計算平均F1分數
        
        Returns:
            float: 平均F1分數
        """
        f1 = self.f1_score()
        return f1.mean().item()
    
    def get_results(self):
        """
        獲取所有評估結果
        
        Returns:
            dict: 評估結果字典
        """
        results = {
            "Pixel_Accuracy": self.pixel_accuracy(),
            "Mean_Accuracy": self.mean_accuracy(),
            "Mean_IoU": self.mean_iou(),
            "Mean_F1": self.mean_f1(),
            "Class_Accuracy": self.class_accuracy().tolist(),
            "Class_IoU": self.class_iou().tolist(),
            "Class_F1": self.f1_score().tolist(),
            "Num_Samples": self.num_samples
        }
        return results