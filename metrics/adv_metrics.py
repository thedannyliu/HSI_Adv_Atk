import torch
import numpy as np


class AdvMetrics:
    """
    對抗攻擊評估指標計算類
    
    計算包括:
    - 攻擊成功率 (Attack Success Rate)
    - 擾動范數 (Perturbation Norm)：L0、L1、L2、L∞
    - 信噪比 (Signal-to-Noise Ratio)
    - 結構相似性 (Structural Similarity)
    """
    
    def __init__(self, device=None):
        """
        初始化
        
        Args:
            device: 計算設備
        """
        self.device = device if device is not None else torch.device('cpu')
        self.total_samples = 0
        self.success_count = 0
        self.l0_norms = []
        self.l1_norms = []
        self.l2_norms = []
        self.linf_norms = []
        self.snrs = []
        self.ssims = []
        
    def _compute_norms(self, original, adversarial):
        """
        計算擾動不同范數的值
        
        Args:
            original: 原始樣本, shape=[B, C, H, W]
            adversarial: 對抗樣本, shape=[B, C, H, W]
            
        Returns:
            tuple: (l0_norm, l1_norm, l2_norm, linf_norm)
        """
        # 計算擾動
        perturbation = adversarial - original
        
        # 計算L0范數 (非零元素的數量)
        l0_norm = torch.count_nonzero(perturbation, dim=(1, 2, 3)).float()
        total_elements = torch.prod(torch.tensor(perturbation.shape[1:]))
        l0_norm = l0_norm / total_elements  # 歸一化
        
        # 計算L1范數
        l1_norm = torch.norm(perturbation, p=1, dim=(1, 2, 3))
        
        # 計算L2范數
        l2_norm = torch.norm(perturbation, p=2, dim=(1, 2, 3))
        
        # 計算L∞范數
        linf_norm = torch.norm(perturbation, p=float('inf'), dim=(1, 2, 3))
        
        return l0_norm, l1_norm, l2_norm, linf_norm
    
    def _compute_snr(self, original, adversarial):
        """
        計算信噪比
        
        Args:
            original: 原始樣本, shape=[B, C, H, W]
            adversarial: 對抗樣本, shape=[B, C, H, W]
            
        Returns:
            torch.Tensor: 信噪比 (dB)
        """
        # 計算擾動
        perturbation = adversarial - original
        
        # 計算信號能量
        signal_power = torch.sum(original ** 2, dim=(1, 2, 3))
        
        # 計算噪聲能量
        noise_power = torch.sum(perturbation ** 2, dim=(1, 2, 3))
        
        # 計算SNR (dB)
        snr = 10 * torch.log10(signal_power / noise_power.clamp(min=1e-10))
        
        return snr
    
    def _compute_ssim(self, original, adversarial):
        """
        計算結構相似性
        
        Args:
            original: 原始樣本, shape=[B, C, H, W]
            adversarial: 對抗樣本, shape=[B, C, H, W]
            
        Returns:
            torch.Tensor: SSIM值
        """
        # SSIM參數
        k1 = 0.01
        k2 = 0.03
        L = 1.0  # 像素值範圍
        
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        
        # 計算各通道的均值和方差
        mu_x = torch.mean(original, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(adversarial, dim=(2, 3), keepdim=True)
        
        sigma_x = torch.var(original, dim=(2, 3), keepdim=True, unbiased=False)
        sigma_y = torch.var(adversarial, dim=(2, 3), keepdim=True, unbiased=False)
        
        sigma_xy = torch.mean((original - mu_x) * (adversarial - mu_y), dim=(2, 3), keepdim=True)
        
        # 計算SSIM
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        # 計算每個樣本的平均SSIM
        ssim = torch.mean(ssim, dim=1).squeeze()
        
        return ssim
    
    def update(self, original_pred, adversarial_pred, original_img, adversarial_img, true_label=None):
        """
        更新指標
        
        Args:
            original_pred: 原始預測, shape=[B, H, W]
            adversarial_pred: 對抗預測, shape=[B, H, W]
            original_img: 原始影像, shape=[B, C, H, W]
            adversarial_img: 對抗影像, shape=[B, C, H, W]
            true_label: 真實標籤, shape=[B, H, W]
        """
        batch_size = original_img.size(0)
        self.total_samples += batch_size
        
        # 計算攻擊成功率
        if true_label is not None:
            # 只計算原始預測正確的樣本
            correct_mask = (original_pred == true_label)
            # 判斷對抗樣本成功與否
            success_mask = correct_mask & (adversarial_pred != true_label)
        else:
            # 如果沒有真實標籤，直接比較兩個預測
            success_mask = (original_pred != adversarial_pred)
        
        # 對每個樣本計算成功與否
        success = torch.any(success_mask, dim=(1, 2))
        self.success_count += torch.sum(success).item()
        
        # 計算範數
        l0_norm, l1_norm, l2_norm, linf_norm = self._compute_norms(original_img, adversarial_img)
        self.l0_norms.extend(l0_norm.cpu().tolist())
        self.l1_norms.extend(l1_norm.cpu().tolist())
        self.l2_norms.extend(l2_norm.cpu().tolist())
        self.linf_norms.extend(linf_norm.cpu().tolist())
        
        # 計算SNR
        snr = self._compute_snr(original_img, adversarial_img)
        self.snrs.extend(snr.cpu().tolist())
        
        # 計算SSIM
        ssim = self._compute_ssim(original_img, adversarial_img)
        self.ssims.extend(ssim.cpu().tolist())
    
    def attack_success_rate(self):
        """
        計算攻擊成功率
        
        Returns:
            float: 攻擊成功率
        """
        if self.total_samples == 0:
            return 0.0
        return self.success_count / self.total_samples
    
    def average_l0_norm(self):
        """
        計算平均L0範數
        
        Returns:
            float: 平均L0範數
        """
        if not self.l0_norms:
            return 0.0
        return np.mean(self.l0_norms)
    
    def average_l1_norm(self):
        """
        計算平均L1範數
        
        Returns:
            float: 平均L1範數
        """
        if not self.l1_norms:
            return 0.0
        return np.mean(self.l1_norms)
    
    def average_l2_norm(self):
        """
        計算平均L2範數
        
        Returns:
            float: 平均L2範數
        """
        if not self.l2_norms:
            return 0.0
        return np.mean(self.l2_norms)
    
    def average_linf_norm(self):
        """
        計算平均L∞範數
        
        Returns:
            float: 平均L∞範數
        """
        if not self.linf_norms:
            return 0.0
        return np.mean(self.linf_norms)
    
    def average_snr(self):
        """
        計算平均信噪比
        
        Returns:
            float: 平均信噪比
        """
        if not self.snrs:
            return 0.0
        return np.mean(self.snrs)
    
    def average_ssim(self):
        """
        計算平均結構相似性
        
        Returns:
            float: 平均結構相似性
        """
        if not self.ssims:
            return 0.0
        return np.mean(self.ssims)
    
    def get_results(self):
        """
        獲取所有評估結果
        
        Returns:
            dict: 評估結果字典
        """
        results = {
            "Attack_Success_Rate": self.attack_success_rate(),
            "Average_L0_Norm": self.average_l0_norm(),
            "Average_L1_Norm": self.average_l1_norm(),
            "Average_L2_Norm": self.average_l2_norm(),
            "Average_Linf_Norm": self.average_linf_norm(),
            "Average_SNR_dB": self.average_snr(),
            "Average_SSIM": self.average_ssim(),
            "Total_Samples": self.total_samples,
            "Success_Count": self.success_count
        }
        return results 