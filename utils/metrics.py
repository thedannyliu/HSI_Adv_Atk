import torch
import numpy as np

class SegMetrics:
    """
    Computes segmentation metrics based on the confusion matrix.
    Matches the implementation in DFCN_Py_all/DFCN_Py/HRNetV2/metrics/metrics.py
    """
    
    def __init__(self, n_class, device):
        self.n_class = n_class
        self.device = device # Store device for resetting
        self.confusion_matrix = torch.zeros((n_class, n_class), dtype=torch.int64).to(device)
        
    def update(self, true, pred):
        """
        Update the confusion matrix.
        Args:
            true: Ground truth labels (B, H, W), LongTensor.
            pred: Predicted labels (B, H, W), LongTensor.
        """
        for t, p in zip(true, pred):
            self.confusion_matrix += self._fast_hist(t.flatten(), p.flatten())
            
    def _fast_hist(self, true, pred):
        """
        Compute the confusion matrix for a flattened label pair.
        Args:
            true: Flattened ground truth labels.
            pred: Flattened predicted labels.
        Returns:
            Confusion matrix update (n_class, n_class).
        """
        mask = (true >= 0) & (true < self.n_class)
        # Ensure indices are within bounds before calculating bin counts
        true_masked = true[mask].to(dtype=torch.int64)
        pred_masked = pred[mask].to(dtype=torch.int64)
        # Calculate index for bincount: label_true * n_class + label_pred
        hist = torch.bincount(
            self.n_class * true_masked + pred_masked, 
            minlength=self.n_class**2
        ).reshape(self.n_class, self.n_class)
        return hist
    
    def get_results(self):
        """
        Compute metrics from the accumulated confusion matrix.
        Returns:
            Dictionary containing overall accuracy, mean accuracy, mean IoU, and class IoU.
        """
        hist = self.confusion_matrix.float() # Use float for calculations
        
        # Overall Accuracy
        acc = torch.diag(hist).sum() / hist.sum()
        
        # Per-Class Accuracy and Mean Accuracy
        acc_cls = torch.diag(hist) / hist.sum(dim=1) # Accuracy for each class
        # Handle cases where a class has no samples (division by zero -> NaN)
        acc_cls_nanmean = torch.nanmean(acc_cls) # Calculate mean ignoring NaNs
        if torch.isnan(acc_cls_nanmean): # If all are NaN (no samples in any class?)
             acc_cls_nanmean = torch.tensor(0.0, device=self.device)
             
        # Per-Class IoU and Mean IoU
        iou = torch.diag(hist) / (hist.sum(dim=0) + hist.sum(dim=1) - torch.diag(hist))
        mean_iou_nanmean = torch.nanmean(iou) # Calculate mean ignoring NaNs
        if torch.isnan(mean_iou_nanmean): # If all are NaN
             mean_iou_nanmean = torch.tensor(0.0, device=self.device)
             
        # Class IoU dictionary
        cls_iou = {i: iou[i].item() if not torch.isnan(iou[i]) else 0.0 for i in range(self.n_class)}
        
        return {
            "Overall acc": acc.item() if not torch.isnan(acc) else 0.0,
            "Mean acc": acc_cls_nanmean.item(),
            "Mean IoU": mean_iou_nanmean.item(),
            "Class IoU": cls_iou
        }
        
    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = torch.zeros((self.n_class, self.n_class), dtype=torch.int64).to(self.device)

# --- Keep existing functions if they are still needed, otherwise they can be removed --- 
# (calculate_miou and calculate_accuracy are effectively replaced by SegMetrics.get_results()['Mean IoU'] and ['Overall acc'])

def calculate_miou(pred, target, debug=False):
    """
    Calculates Mean Intersection over Union (mIoU) for binary case.
    Kept for potential compatibility, but SegMetrics is preferred.
    Args:
        pred (torch.Tensor): Predictions (B, H, W), Long or Float.
        target (torch.Tensor): Ground truth (B, H, W), Long.
    Returns:
        float: Mean IoU.
    """
    # Use SegMetrics for consistent calculation
    metric = SegMetrics(n_class=2, device=pred.device) 
    pred_long = (pred > 0.5).long() if pred.is_floating_point() else pred.long()
    metric.update(target.long(), pred_long)
    results = metric.get_results()
    return results["Mean IoU"] 

def calculate_accuracy(pred, target, debug=False):
    """
    Calculates pixel accuracy for binary case.
    Kept for potential compatibility, but SegMetrics is preferred.
     Args:
        pred (torch.Tensor): Predictions (B, H, W), Long or Float.
        target (torch.Tensor): Ground truth (B, H, W), Long.
    Returns:
        float: Accuracy.
    """
    # Use SegMetrics for consistent calculation
    metric = SegMetrics(n_class=2, device=pred.device) 
    pred_long = (pred > 0.5).long() if pred.is_floating_point() else pred.long()
    metric.update(target.long(), pred_long)
    results = metric.get_results()
    return results["Overall acc"] 