import random
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

def extract_high_frequency_with_maxpool(in_feature_map, kernel_size=2, device='cuda'):
    # 将输入特征图移到指定设备（如GPU）
    in_feature_map = in_feature_map.to(device)
    
    # Step 1: Downsample the input feature map using max pooling
    temp1 = F.max_pool2d(in_feature_map, kernel_size=kernel_size, stride=kernel_size)
    
    # Step 2: Upsample back to the original size using bilinear interpolation
    temp2 = F.interpolate(temp1, size=in_feature_map.shape[2:], mode='bilinear', align_corners=False)
    
    # Step 3: Subtract to get the high-frequency information
    high_frequency_info = in_feature_map - temp2
    
    return high_frequency_info

class HighFrequencyLoss(nn.Module):
    def __init__(self):
        super(HighFrequencyLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        

    def forward(self, real, fake):
        real_vgg, fake_vgg = self.vgg(real), self.vgg(fake)
        loss = 0
        real_HF = []
        fake_HF = []
        for i in range(len(real_vgg)):
            real_HF.append(extract_high_frequency_with_maxpool(real_vgg[i]))
            fake_HF.append(extract_high_frequency_with_maxpool(fake_vgg[i]))
        for i in range(len(real_HF)):
            loss = loss + self.weights[i] * self.l1(real_HF[i], fake_HF[i])

        return loss


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss = loss + self.weights[i] * contrastive
        return loss
