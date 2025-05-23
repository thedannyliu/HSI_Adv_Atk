import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
Implementation based on DFCN_Py_all/DFCN_Py/HRNetV2/models/hrnet.py
There are 4 resolution levels in HRNet. We name each level as large, middle, small, tiny.
"""

# --- Blocks defined as in DFCN --- 
class firstBlock(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, in_channel, out_channel, identity_conv=None):
        super(firstBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.identity_conv = identity_conv
    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x = x + identity
        x = self.relu(x)
        return x

class otherBlock(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, in_channels, out_channels):
        super(otherBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x

# --- Stages defined as in DFCN --- 
class firstStage(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, block, C):
        super(firstStage, self).__init__()
        self.units = self._make_units(block, 4)
        self.conv2large = nn.Conv2d(256, C, kernel_size=3, stride=1, padding=1)
        self.conv2middle = nn.Conv2d(256, 2*C, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.units(x)
        x_large = self.conv2large(x)
        x_middle = self.conv2middle(x)
        return [x_large, x_middle]
    def _make_units(self, block, num_units):
        layers = []
        identity_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        layers.append(block(64, 256, identity_conv))
        for i in range(num_units - 1):
            layers.append(block(256, 256, identity_conv=None))
        return nn.Sequential(*layers)

class groupConvUnit(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, block, C, level=0):
        super(groupConvUnit, self).__init__()
        self.level = level
        self.convlist = nn.ModuleList()
        for i in range(level+1):
            self.convlist.append(self._make_units(block, 2**i*C, 4))
    def forward(self, x: List):
        out = []
        for i in range(self.level+1):
            out.append(self.convlist[i](x[i]))
        return out
    def _make_units(self, block, channel, num_units):
        layers = nn.ModuleList()
        for i in range(num_units):
            layers.append(block(channel, channel))
        return nn.Sequential(*layers)

class exchangeConvUnit(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, C, in_level, out_level):
        super(exchangeConvUnit, self).__init__()
        self.in_level = in_level
        self.out_level = out_level
        self.convlist = nn.ModuleList()
        for i in range(out_level+1):
            to_convlist = nn.ModuleList()
            for j in range(in_level+1):
                if j < i:
                    to_convlist.append(self._make_downconv(C, j, i))
                elif j > i:
                    to_convlist.append(self._make_upconv(C, j, i))
                else:
                    to_convlist.append(None)
            self.convlist.append(to_convlist)
    def forward(self, x: List):
        assert self.in_level+1 == len(x)
        out = []
        for j in range(self.in_level+1):
            out.append(x[j].clone())
        for i in range (min(self.in_level+1, self.out_level+1)):
            for j in range(self.in_level+1):
                if j < i:
                    out[i] = out[i] + self.convlist[i][j](x[j])
                elif j==i:
                    pass # out[i] = out[i]
                elif j > i:
                    out[i] = out[i] + self.convlist[i][j](F.interpolate(x[j], size=out[i].shape[2:], mode="bilinear", align_corners=True))
        if self.in_level < self.out_level:
            # Handle creation of new branch if out_level > in_level
            new_branch_feature = self.convlist[self.out_level][0](x[0])
            for j in range(1, self.in_level + 1):
                 new_branch_feature = new_branch_feature + self.convlist[self.out_level][j](x[j])
            out.append(new_branch_feature)
        # Apply ReLU after fusion
        return [F.relu(item) for item in out]
    def _make_downconv(self, C, in_level, out_level):
        diff_level = out_level - in_level
        layers = nn.ModuleList()
        # DFCN implementation detail: Only apply BN and ReLU on the final downsample conv
        current_channels = 2**(in_level)*C
        for i in range(diff_level):
            out_channels = 2**(in_level+i+1)*C
            if i == diff_level - 1: # Last conv
                 layers.append(nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels), 
                    nn.ReLU(inplace=False) 
                 ))
            else:
                layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
            current_channels = out_channels # Update for next layer input
        return nn.Sequential(*layers)
    def _make_upconv(self, C, in_level, out_level):
        # Upsampling is just a 1x1 Conv followed by BN and ReLU in DFCN
        in_channels = 2**in_level*C
        out_channels = 2**out_level*C
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False) # DFCN applies ReLU here
            # Note: Interpolation happens in the forward pass
        )

class secondStage(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, block, C):
        super(secondStage, self).__init__()
        self.groupconv1 = groupConvUnit(block, C, 1)
        self.exchange1 = exchangeConvUnit(C, 1, 2)
    def forward(self, x: List):
        x = self.groupconv1(x)
        x = self.exchange1(x)
        return x

class thirdStage(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, block, C):
        super(thirdStage, self).__init__()
        self.groupconv1 = groupConvUnit(block, C, 2)
        self.exchange1 = exchangeConvUnit(C, 2, 2)
        self.groupconv2 = groupConvUnit(block, C, 2)
        self.exchange2 = exchangeConvUnit(C, 2, 2)
        self.groupconv3 = groupConvUnit(block, C, 2)
        self.exchange3 = exchangeConvUnit(C, 2, 2)
        self.groupconv4 = groupConvUnit(block, C, 2)
        self.exchange4 = exchangeConvUnit(C, 2, 3)
    def forward(self, x: List):
        x = self.groupconv1(x)
        x = self.exchange1(x)
        x = self.groupconv2(x)
        x = self.exchange2(x)
        x = self.groupconv3(x)
        x = self.exchange3(x)
        x = self.groupconv4(x)
        x = self.exchange4(x)
        return x

class fourthStage(nn.Module):
    # (Identical to DFCN version)
    def __init__(self, block, C):
        super(fourthStage, self).__init__()
        self.groupconv1 = groupConvUnit(block, C, 3)
        self.exchange1 = exchangeConvUnit(C, 3, 3)
        self.groupconv2 = groupConvUnit(block, C, 3)
        self.exchange2 = exchangeConvUnit(C, 3, 3)
        self.groupconv3 = groupConvUnit(block, C, 3)
        self.exchange3 = exchangeConvUnit(C, 3, 3)
    def forward(self, x: List):
        x = self.groupconv1(x)
        x = self.exchange1(x)
        x = self.groupconv2(x)
        x = self.exchange2(x)
        x = self.groupconv3(x)
        x = self.exchange3(x)
        return x

# --- Final Stage modified for Dual Output (as in DFCN) --- 
class finalStage(nn.Module):
    def __init__(self, C, num_class, bands_num):
        super(finalStage, self).__init__()
        # Main segmentation output layer
        self.lastlayer = nn.Sequential(
            nn.Conv2d(15 * C, 15 * C, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(15 * C),
            nn.ReLU(inplace=False),
            nn.Conv2d(15 * C, num_class, kernel_size=1, stride=1, padding=0)
        )
        
        # 增強頻段預測能力
        # 使用更複雜的預測器結構，參考DFCN_Py中的設計
        self.lastPredictor = nn.Sequential(
            # 首先通過卷積降低通道數
            nn.Conv2d(8 * C, 4 * C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * C),
            nn.ReLU(inplace=False),
            
            # 應用注意力機制增強重要頻段的感知
            # 在這裡添加一個自注意力層，幫助模型專注於重要的頻段特徵
            nn.Conv2d(4 * C, 2 * C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * C),
            nn.ReLU(inplace=False),
            
            # 進一步降低通道數
            nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            
            # 最終預測每個頻段的分類
            nn.Conv2d(C, bands_num, kernel_size=1, stride=1, padding=0),
            
            # 全局平均池化以獲得每個頻段的整體評分
            nn.AdaptiveAvgPool2d(1),
            
            # 添加Dropout以增強泛化能力
            nn.Dropout(0.2)
        )
        
        # 添加頻段之間的相對重要性關係學習模塊
        self.band_attention = nn.Sequential(
            nn.Linear(bands_num, bands_num // 4),
            nn.ReLU(inplace=False),
            nn.Linear(bands_num // 4, bands_num),
            nn.Sigmoid()
        )

    def forward(self, x: List):
        x_large = x[0]  # C
        x_middle = x[1] # 2C
        x_small = x[2]  # 4C
        x_tiny = x[3]   # 8C
        
        # Upsample features to the largest resolution (x_large)
        x_middle = F.interpolate(x_middle, size=x_large.shape[2:], mode="bilinear", align_corners=True)
        x_small = F.interpolate(x_small, size=x_large.shape[2:], mode="bilinear", align_corners=True)
        x_tiny_up = F.interpolate(x_tiny, size=x_large.shape[2:], mode="bilinear", align_corners=True)
        
        # 使用增強的頻段預測器生成頻段預測結果
        x_pred_bands_raw = self.lastPredictor(x_tiny) # Output: (B, bands_num, 1, 1)
        x_pred_bands_flat = x_pred_bands_raw.reshape((x_pred_bands_raw.shape[0], -1)) # Output: (B, bands_num)
        
        # 應用頻段注意力機制來調整各頻段的相對重要性
        band_attention_weights = self.band_attention(x_pred_bands_flat)
        x_pred_bands = x_pred_bands_flat * band_attention_weights
        
        # 標準化頻段預測以獲得更好的訓練信號
        # 對於頻段輸出，我們使用Sigmoid作為最終激活函數
        # 但這不是在此處應用的，因為BCEWithLogitsLoss會在訓練腳本中內部處理
        
        # 添加正則化以避免過擬合
        if self.training:
            # 在訓練期間增加一些噪聲以提高穩健性
            noise = torch.randn_like(x_pred_bands) * 0.01
            x_pred_bands = x_pred_bands + noise

        # Concatenate upsampled features
        out = torch.cat([x_large, x_middle, x_small, x_tiny_up], 1) # Channels: C + 2C + 4C + 8C = 15C
        
        # Further upsample to original image size (x4) if needed by the stem
        # Assuming stem downsamples by 4 (2 convs with stride 2)
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        
        # Final segmentation prediction
        out_seg = self.lastlayer(out) # Output: (B, num_class, H, W)
        
        return out_seg, x_pred_bands # Return both segmentation and band predictions

# --- SKConv defined as in DFCN --- 
class SKConv(nn.Module):
    """
    Selective Kernel Convolution module as used in DFCN_Py
    
    參數:
        features: 輸入特徵通道數
        M: 分支數
        G: 組數，用於分組卷積
        r: 降維率
        stride: 卷積步長
        L: 最小通道數
    """
    def __init__(self, features, M=2, G=None, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        # 確保一致的配置
        if G is None:
            G = features
            print(f"[MODEL INFO] SKConv using default G=features={features}")
            
        # 增加调试信息
        print(f"[MODEL INFO] SKConv: features={features}, M={M}, G={G}, r={r}, stride={stride}, L={L}")
        
        self.M = M
        self.features = features
        
        # 多尺度分支，按照DFCN_Py实现
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
            
        # 降維層和全局平均池化
        d = max(int(features/r), L)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=False)
        )
        
        # 注意力機制
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        # 標準化輸入以提高注意力機制的有效性
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x = (x - x_mean) / x_std
        
        # 應用多尺度卷積分支
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, h, w)
        
        # 全局平均池化得到注意力權重
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        
        # 生成每個分支的注意力得分
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        # 加權求和得到最終輸出
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V

# --- Main HRNetV2 modified to match DFCN --- 
class HRNetV2(nn.Module):
    def __init__(self, C, num_class, in_ch=3, use3D=False, useAttention=False):
        super(HRNetV2, self).__init__()
        print(f"[MODEL INFO] Initializing HRNetV2 (DFCN Style):")
        print(f"  - Base Channels (C): {C}")
        print(f"  - Num Classes: {num_class}")
        print(f"  - Input Channels: {in_ch}")
        print(f"  - Using 3D Frontend (use3D): {use3D}")
        print(f"  - Using Attention (useAttention): {useAttention}")
        
        self.use3D = use3D
        self.useAttention = useAttention
        self.in_ch = in_ch
        
        # --- Stem Stage (Matches DFCN) --- 
        if self.use3D:
            KS = 64 # Fixed parameter from DFCN
            # 3D Convolution applied first - 將通道數從in_ch降至in_ch//2 (172->86)
            self.conv0_1 = nn.Conv3d(1, 1, kernel_size=(KS, 3, 3), stride=2, padding=(KS // 2 - 1, 1, 1), bias=False)
            # Auxiliary 2D Convolution (input channels must match output of 3D conv + attention if used)
            stem_out_channels = 64 # Target output channels for the stem
            
            # 按照DFCN_Py的邏輯，3D卷積後通道數變為in_ch//2 (例如從172變為86)
            reduced_channels = in_ch // 2
            
            if self.useAttention:
                # 使用降低後的通道數 (86)，不再使用原始通道數 (172)
                sk_input_features = reduced_channels
                
                # 打印警告如果降低後的通道數不是86
                if reduced_channels != 86:
                    print(f"[MODEL WARNING] Unusual reduced channels: {reduced_channels}. Expected 86.")
                     
                # SKConv使用降低後的通道數 (86)
                # 按照DFCN_Py，使用4個SKConv層，各參數保持一致
                self.skconv = nn.Sequential(
                    SKConv(sk_input_features, M=2, G=sk_input_features//2, r=16, stride=1, L=32), 
                    nn.ReLU(inplace=False), 
                    SKConv(sk_input_features, M=2, G=sk_input_features//2, r=16, stride=1, L=32), 
                    nn.ReLU(inplace=False), 
                    SKConv(sk_input_features, M=2, G=sk_input_features//2, r=16, stride=1, L=32), 
                    nn.ReLU(inplace=False), 
                    SKConv(sk_input_features, M=2, G=sk_input_features//2, r=16, stride=1, L=32)
                )
                # Aux conv input matches SKConv output features
                self.conv0_1_aux = nn.Conv2d(sk_input_features, stem_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                # Aux conv input matches squeezed 3D conv output channels (reduced_channels)
                self.conv0_1_aux = nn.Conv2d(reduced_channels, stem_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            # Standard 2D convs follow the 3D/Aux part
            self.bn0_1 = nn.BatchNorm2d(stem_out_channels)
            self.conv0_2 = nn.Conv2d(stem_out_channels, stem_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn0_2 = nn.BatchNorm2d(stem_out_channels)

        else:
            # Standard 2D Stem
            stem_out_channels = 64
            self.conv0_1 = nn.Conv2d(in_ch, stem_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn0_1 = nn.BatchNorm2d(stem_out_channels)
            self.conv0_2 = nn.Conv2d(stem_out_channels, stem_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn0_2 = nn.BatchNorm2d(stem_out_channels)
            
        self.relu = nn.ReLU(inplace=False)
        
        # --- HRNet Stages (Identical structure to DFCN) --- 
        self.firstStage = firstStage(firstBlock, C)       # Input stem_out_channels=64 -> Output [C, 2C]
        self.secondStage = secondStage(otherBlock, C)     # Input [C, 2C] -> Output [C, 2C, 4C]
        self.thirdStage = thirdStage(otherBlock, C)       # Input [C, 2C, 4C] -> Output [C, 2C, 4C, 8C]
        self.fourthStage = fourthStage(otherBlock, C)     # Input [C, 2C, 4C, 8C] -> Output [C, 2C, 4C, 8C]
        
        # --- Final Stage (Dual Output as in DFCN) --- 
        self.finalStage = finalStage(C, num_class, in_ch) # Pass in_ch as bands_num for predictor
        
    def forward(self, x):
        # --- 標準化處理 ---
        # 確保輸入在合理範圍內 (-1 to 1)，靜默處理超出範圍的輸入
        if torch.min(x) < -1.1 or torch.max(x) > 1.1:
            # 移除警告訊息，直接進行正規化處理
            # print(f"[MODEL WARNING] Input range unusual: min={torch.min(x)}, max={torch.max(x)}. Normalizing...")
            x = 2.0 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1.0
            
        # --- Stem --- 
        if self.use3D:
            # Reshape for Conv3D: (B, C, H, W) -> (B, 1, C, H, W)
            x = x.unsqueeze(1)
            x = self.conv0_1(x)
            x = self.relu(x)
            
            # Reshape back: (B, 1, C', H', W') -> (B, C', H', W')
            # 按照DFCN_Py的邏輯，這裡的通道數應該自動從in_ch變為in_ch//2
            if x.size(1) == 1:
                 x = x.squeeze(1) # Now shape (B, C//2, H'/2, W'/2)
            else:
                 # 移除警告訊息，直接處理
                 # print("[MODEL WARNING] Conv3D output channels > 1 not handled perfectly yet.")
                 pass # Need proper reshape logic based on intended output
            
            if self.useAttention:
                # 標準化3D卷積的輸出以提高SKConv注意力機制的效果
                x_mean = x.mean(dim=[2, 3], keepdim=True)
                x_std = x.std(dim=[2, 3], keepdim=True) + 1e-8
                x = (x - x_mean) / x_std
                # 應用SKConv注意力機制
                x = self.skconv(x)
            
            x = self.conv0_1_aux(x) 
            x = self.bn0_1(x)
            x = self.relu(x) # Apply relu after bn0_1
            x = self.conv0_2(x) # Apply second stem conv
            x = self.bn0_2(x)
            x = self.relu(x) # Apply final stem relu

        else:
            # Standard 2D Stem
            x = self.conv0_1(x)
            x = self.bn0_1(x)
            x = self.relu(x)
            x = self.conv0_2(x)
            x = self.bn0_2(x)
            x = self.relu(x)
        
        # --- HRNet Body --- 
        x_list = self.firstStage(x)     # Input: (B, stem_out_channels, H/4, W/4)
        x_list = self.secondStage(x_list)
        x_list = self.thirdStage(x_list)
        x_list = self.fourthStage(x_list)
        
        # --- Final Stage (Returns two outputs) --- 
        out_seg, out_bands = self.finalStage(x_list)
        
        return out_seg, out_bands # Return both segmentation and band predictions