import torch
import numpy as np
from sklearn.metrics import jaccard_score, confusion_matrix

class SegMetrics:
    """
    語義分割評估指標
    
    Args:
        n_class (int): 類別數量
        device: 計算設備
    """
    def __init__(self, n_class, device):
        self.n_class = n_class
        self.device = device
        self.confusion_matrix = torch.zeros((n_class, n_class), device=device)

    def update(self, true, pred):
        """
        更新混淆矩陣
        
        Args:
            true (torch.Tensor): 真實標籤
            pred (torch.Tensor): 預測標籤
        """
        for t, p in zip(true, pred):
            self.confusion_matrix += self._fast_hist(t.view(-1), p.view(-1))

    def _fast_hist(self, true, pred):
        """
        快速計算混淆矩陣
        
        Args:
            true (torch.Tensor): 真實標籤
            pred (torch.Tensor): 預測標籤
            
        Returns:
            torch.Tensor: 混淆矩陣
        """
        mask = (true >= 0) & (true < self.n_class)
        hist = torch.bincount(
            self.n_class * true[mask].to(torch.int64) + pred[mask],
            minlength=self.n_class ** 2
        ).reshape(self.n_class, self.n_class)
        return hist

    def get_results(self):
        """
        獲取評估結果
        
        Returns:
            dict: 包含多種評估指標的字典
        """
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        acc_cls = torch.nanmean(acc_cls)
        iou = torch.diag(hist) / (hist.sum(dim=0) + hist.sum(dim=1) - torch.diag(hist))
        mean_iou = torch.nanmean(iou)
        
        # 計算 mAP
        # 對於每一類，計算 Precision 和 Recall
        precision = torch.zeros(self.n_class, device=self.device)
        recall = torch.zeros(self.n_class, device=self.device)
        average_precisions = torch.zeros(self.n_class, device=self.device)
        
        for cls in range(self.n_class):
            tp = hist[cls, cls]
            fp = hist[:, cls].sum() - tp
            fn = hist[cls, :].sum() - tp

            precision[cls] = tp / (tp + fp + 1e-10)
            recall[cls] = tp / (tp + fn + 1e-10)

            # 計算每類的 AP，這裡使用簡單的方式
            average_precisions[cls] = precision[cls]

        mean_ap = average_precisions.mean()
        
        # 計算F1分數
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        mean_f1 = torch.nanmean(f1_scores)

        return {
            "Overall Acc": acc.item(),
            "Mean Acc": acc_cls.item(),
            "Class IoU": iou.cpu().tolist(),
            "Mean IoU": mean_iou.item(),
            "Class AP": precision.cpu().tolist(),
            "Mean AP": mean_ap.item(),
            "Class F1": f1_scores.cpu().tolist(),
            "Mean F1": mean_f1.item(),
        }

    def reset(self):
        """
        重置混淆矩陣
        """
        self.confusion_matrix = torch.zeros((self.n_class, self.n_class), device=self.device)

class AdvMetrics:
    """
    對抗攻擊評估指標
    
    Args:
        device: 計算設備
    """
    def __init__(self, device):
        self.device = device
        self.orig_preds = []
        self.adv_preds = []
        self.orig_imgs = []
        self.adv_imgs = []
        self.labels = []
    
    def update(self, orig_pred, adv_pred, orig_img, adv_img, label):
        """
        更新評估數據
        
        Args:
            orig_pred (torch.Tensor): 原始圖像的預測結果
            adv_pred (torch.Tensor): 對抗樣本的預測結果
            orig_img (torch.Tensor): 原始圖像
            adv_img (torch.Tensor): 對抗樣本
            label (torch.Tensor): 真實標籤
        """
        self.orig_preds.append(orig_pred.detach().cpu())
        self.adv_preds.append(adv_pred.detach().cpu())
        self.orig_imgs.append(orig_img.detach().cpu())
        self.adv_imgs.append(adv_img.detach().cpu())
        self.labels.append(label.detach().cpu())
    
    def get_results(self):
        """
        獲取評估結果
        
        Returns:
            dict: 包含多種評估指標的字典
        """
        orig_preds = torch.cat(self.orig_preds, 0)
        adv_preds = torch.cat(self.adv_preds, 0)
        orig_imgs = torch.cat(self.orig_imgs, 0)
        adv_imgs = torch.cat(self.adv_imgs, 0)
        labels = torch.cat(self.labels, 0)
        
        # 計算攻擊成功率(ASR)
        orig_correct = (orig_preds == labels).float().mean()
        adv_incorrect = (adv_preds != labels).float().mean()
        asr = adv_incorrect / (orig_correct + 1e-10)
        
        # 計算L2擾動
        l2_perturbation = torch.norm(adv_imgs - orig_imgs, p=2, dim=(1,2,3)).mean()
        
        # 計算L∞擾動
        linf_perturbation = torch.max(torch.abs(adv_imgs - orig_imgs), dim=(1,2,3))[0].mean()
        
        # 計算SSIM（需要額外函數）
        ssim_value = self._compute_ssim(orig_imgs, adv_imgs)
        
        return {
            "Attack Success Rate": asr.item(),
            "L2 Perturbation": l2_perturbation.item(),
            "Linf Perturbation": linf_perturbation.item(),
            "SSIM": ssim_value,
        }
    
    def _compute_ssim(self, img1, img2):
        """
        計算結構相似度(SSIM)
        
        Args:
            img1 (torch.Tensor): 圖像1
            img2 (torch.Tensor): 圖像2
            
        Returns:
            float: SSIM值
        """
        # 簡單實現，實際中可使用庫函數
        # 這裡僅返回一個佔位值
        return 1.0 - torch.mean(torch.abs(img1 - img2)).item()
    
    def reset(self):
        """
        重置評估數據
        """
        self.orig_preds = []
        self.adv_preds = []
        self.orig_imgs = []
        self.adv_imgs = []
        self.labels = [] 