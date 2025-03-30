import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

"""
There are 4 resolution levels in HRNet. We name each level as large, middle, small, tiny.
"""

class firstBlock(nn.Module):
    def __init__(self, in_channel, out_channel, identity_conv=None):
        super(firstBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU(inplace=True)
        
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
        
        # add 1x1conv to match number of channels
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity
        x = self.relu(x)
        
        return x
    
class firstStage(nn.Module):
    def __init__(self, block, C):
        super(firstStage, self).__init__()
        
        self.units = self._make_units(block, 4)
        self.conv2large = nn.Conv2d(256, C, kernel_size=3, stride=1, padding=1)
        self.conv2middle = nn.Conv2d(256, 2*C, kernel_size=3, stride=2, padding=1)
#         self.conv2middle1 = SKConv(256, G=256//4)

    def forward(self, x):
        x = self.units(x)
        x_large = self.conv2large(x)
        x_middle = self.conv2middle(x)
        
        return [x_large, x_middle]
        
    def _make_units(self, block, num_units):
        layers = []
        
        # 1st unit
        identity_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        layers.append(block(64, 256, identity_conv))
        
        # 2~num_units units
        for i in range(num_units - 1):
            layers.append(block(256, 256, identity_conv=None))
        
        return nn.Sequential(*layers)


class otherBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(otherBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += identity
        x = self.relu(x)
        
        return x
        
class groupConvUnit(nn.Module):
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
                    out[i] += self.convlist[i][j](x[j])
                elif j==i:
                    out[i] = out[i]
                elif j > i:
                    out[i] += self.convlist[i][j](F.interpolate(x[j], out[i].shape[2:], mode="bilinear", align_corners=True))
        if self.in_level < self.out_level:
            out.append(self.convlist[self.out_level][0](x[0]))
            for j in range(1,self.in_level+1):
                out[self.out_level] += self.convlist[self.out_level][j](x[j])
                
        return out
                
    def _make_downconv(self, C, in_level, out_level):
        diff_level = out_level - in_level
        layers = nn.ModuleList()
        for i in range(diff_level):
            layers.append(nn.Conv2d(2**(in_level+i)*C, 2**(in_level+i+1)*C, kernel_size=3, stride=2, padding=1))
        if diff_level > 1:
            return nn.Sequential(*layers)
        else:
            return layers[0]
    
    def _make_upconv(self, C, in_level, out_level):
        return nn.Conv2d(2**in_level*C, 2**out_level*C, kernel_size=1, stride=1, padding=0)
        
class secondStage(nn.Module):
    def __init__(self, block, C):
        super(secondStage, self).__init__()
        self.groupconv1 = groupConvUnit(block, C, 1)
        self.exchange1 = exchangeConvUnit(C, 1, 2)
    
    def forward(self, x: List):
        x = self.groupconv1(x)
        x = self.exchange1(x)
        
        return x
    
class thirdStage(nn.Module):
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

class finalStage(nn.Module):
    def __init__(self, C, num_class,  bands_num):
        super(finalStage,self).__init__()
        self.lastlayer =nn.Sequential(
            nn.Conv2d(15*C, 15*C, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(15*C),
            nn.ReLU(inplace=True),
            nn.Conv2d(15*C, num_class, kernel_size=1, stride=1, padding=0)
        )
        self.lastPredictor = nn.Sequential(
                            nn.Conv2d(8*C, C, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(C),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(C, bands_num, kernel_size=1, stride=1, padding=0),
                            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x: List):
        x_large = x[0]
        x_middle = x[1]
        x_small = x[2]
        x_tiny = x[3]
        
        x_middle = F.interpolate(x_middle, x_large.shape[2:], mode="bilinear", align_corners=True)
        x_small = F.interpolate(x_small, x_large.shape[2:], mode="bilinear", align_corners=True)
        x_tiny = F.interpolate(x_tiny, x_large.shape[2:], mode="bilinear", align_corners=True)
    
        x_pred = self.lastPredictor(x_tiny)
        x_pred = x_pred.reshape((x_pred.shape[0], -1))
        
        out = torch.cat([x_large, x_middle, x_small, x_tiny],1)
        out = F.interpolate(out, (out.shape[2]*4, out.shape[3]*4), mode="bilinear", align_corners=True)
        out = self.lastlayer(out)
        
        return out, x_pred
    
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=86//2, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V

    
class HRNetV2(nn.Module):
    def __init__(self, C, num_class, in_ch = 3, use3D=False, useAttention=False):
        super(HRNetV2, self).__init__()
        
        # stem stage
        self.use3D = use3D
        self.useAttention = useAttention
        print(use3D)
        if use3D:
            KS = 64
            self.conv0_1 = nn.Conv3d(1, 1, kernel_size=(KS,3,3), stride=2, padding=(KS//2-1,1,1), bias=False)
            self.conv0_1_aux = nn.Conv2d(86, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            if useAttention:
#                 self.gap1 = nn.AdaptiveAvgPool2d(1)
                self.skconv = nn.Sequential(SKConv(86), nn.ReLU(), SKConv(86), nn.ReLU(), SKConv(86), nn.ReLU(), SKConv(86))
            
        else:
            self.conv0_1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False)
            
        self.bn0_1 = nn.BatchNorm2d(64)
        self.conv0_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 1st stage
        self.firstStage = firstStage(firstBlock, C)
        
        # 2nd stage
        self.secondStage = secondStage(otherBlock, C)
        
        # 3rd stage
        self.thirdStage = thirdStage(otherBlock, C)
        
        # 4th stage
        self.fourthStage = fourthStage(otherBlock, C)
        
        #final
        self.finalStage = finalStage(C, num_class, in_ch)
        
    def forward(self, x):
        if self.use3D:
#             print(x.shape)
            x = x.unsqueeze(1)
#             print(x.shape)
            x = self.conv0_1(x)
            x = self.relu(x)
            x = x.squeeze(1)
            
            if self.useAttention:
                x = self.skconv(x)
            
            x = self.conv0_1_aux(x) 



        else:
            x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        
        x_list = self.firstStage(x)
        x_list = self.secondStage(x_list)
        x_list = self.thirdStage(x_list)
        x_list = self.fourthStage(x_list)
        out = self.finalStage(x_list)
        
        return out