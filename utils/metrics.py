import torch
import numpy as np

def calculate_miou(pred, target, debug=False):
    """
    計算平均交並比 (Mean Intersection over Union)
    
    Args:
        pred (torch.Tensor): 預測結果，形狀為 (B, H, W)
        target (torch.Tensor): 標籤，形狀為 (B, H, W)
        debug (bool): 是否輸出調試信息
    
    Returns:
        float: 平均交並比
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
    
    # 確保形狀一致
    if pred.dim() == 4 and pred.size(1) > 1:  # (B, C, H, W) 格式
        pred = torch.argmax(pred, dim=1)  # 轉換為 (B, H, W)
    
    # 轉換為二值張量
    pred = (pred > 0).float()
    target = (target > 0).float()
    
    if debug:
        print(f"Pred shape: {pred.shape}, unique values: {torch.unique(pred)}")
        print(f"Target shape: {target.shape}, unique values: {torch.unique(target)}")
    
    # 計算各類別的 IoU
    intersection = (pred * target).sum((1, 2))  # 交集：兩者都為1的像素
    union = pred.sum((1, 2)) + target.sum((1, 2)) - intersection  # 聯集：去除重複計算的部分
    
    # 避免除以零
    valid = union > 0
    
    if debug:
        print(f"Intersection: {intersection}")
        print(f"Union: {union}")
        print(f"Valid: {valid}")
    
    # 計算 IoU
    iou = torch.zeros_like(union)
    iou[valid] = intersection[valid] / union[valid]
    
    # 計算平均 IoU
    miou = iou.mean().item()
    
    if debug:
        print(f"IoU: {iou}")
        print(f"Mean IoU: {miou}")
    
    return miou

def calculate_accuracy(pred, target, debug=False):
    """
    計算準確率
    
    Args:
        pred (torch.Tensor): 預測結果，形狀為 (B, H, W) 或 (B, C, H, W)
        target (torch.Tensor): 標籤，形狀為 (B, H, W)
        debug (bool): 是否輸出調試信息
    
    Returns:
        float: 準確率
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)
    
    # 確保形狀一致
    if pred.dim() == 4 and pred.size(1) > 1:  # (B, C, H, W) 格式
        pred = torch.argmax(pred, dim=1)  # 轉換為 (B, H, W)
    
    # 確保數據類型一致
    pred = (pred > 0).long()
    target = (target > 0).long()
    
    if debug:
        print(f"Pred shape: {pred.shape}, unique values: {torch.unique(pred, return_counts=True)}")
        print(f"Target shape: {target.shape}, unique values: {torch.unique(target, return_counts=True)}")
    
    # 計算準確率
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    accuracy = correct / total
    
    if debug:
        print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy}")
    
    return accuracy.item() 