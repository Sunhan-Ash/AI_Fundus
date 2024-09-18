import random
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

def fourier_low_pass(image, cutoff=30):
    """ 使用傅里叶变换实现低通滤波器 """
    # 只对绿色通道进行傅里叶变换
    green_channel = image[:, 1:2, :, :]
    
    # 进行 2D 傅里叶变换
    fft = torch.fft.fft2(green_channel)
    fft_shift = torch.fft.fftshift(fft)

    # 创建低通滤波器掩膜
    rows, cols = green_channel.shape[2:]
    crow, ccol = rows // 2 , cols // 2

    mask = torch.zeros_like(green_channel, dtype=torch.complex64)
    mask[:, :, crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1

    # 应用掩膜
    fft_shift = fft_shift * mask

    # 逆傅里叶变换回时域
    fft_ishift = torch.fft.ifftshift(fft_shift)
    low_freq = torch.fft.ifft2(fft_ishift).real

    return low_freq


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """ 使用高斯模糊提取低频信息 """
    # 创建高斯核
    def get_gaussian_kernel(kernel_size, sigma):
        ax = torch.arange(-(kernel_size - 1) // 2 + 1, (kernel_size - 1) // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid([ax, ax], indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    # 获取高斯核并调整形状为 [1, 1, kernel_size, kernel_size]
    kernel = get_gaussian_kernel(kernel_size, sigma)
    
    # 如果图像是多通道的，处理绿色通道
    # green_channel = image[:, 1:2, :, :]  # 选择绿色通道，假设 image 是 [B, C, H, W] 形式
    
    # 调整高斯核的形状
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(image.device)  # 适配卷积
    low_freq = F.conv2d(image, kernel, padding=kernel_size // 2, groups=1)  # 单通道卷积
    return low_freq

def extract_high_frequency(image, kernel_size=5, sigma=1.0):
    """ 提取高频信息：原图减去低频信息 """
    low_freq = gaussian_blur(image, kernel_size, sigma)
    # low_freq = fourier_low_pass(image=image)
    high_freq = image - low_freq  # 高频信息 = 原图 - 低频信息
    return high_freq

class HF_loss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(HF_loss, self).__init__()
        self.loss_function = nn.L1Loss().cuda()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, input, enhance):
        loss = 0
        # 提取绿色通道 (索引为1的通道)
        original_green_channel = input[:, 1:2, :, :]  # 只提取绿色通道
        enhanced_green_channel = enhance[:, 1:2, :, :]
        # 提取高频信息
        original_high_freq = extract_high_frequency(original_green_channel, self.kernel_size, self.sigma)
        enhanced_high_freq = extract_high_frequency(enhanced_green_channel, self.kernel_size, self.sigma)
        loss = self.loss_function(original_high_freq, enhanced_high_freq)
        return loss



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
