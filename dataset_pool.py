
import random
import cv2
import numpy as np
import os
from glob import glob
# 定义新的目标目录
hazy_dir_base = './dataset/hazy/'
gt_dir = './dataset/GT/'

# 创建新的目标目录
source_dir = './dataset/diabetic-retinopathy-detection/quality_0'
mask_dir = './dataset/diabetic-retinopathy-detection/mask'
os.makedirs(hazy_dir_base, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# 获取所有图片路径
image_paths = glob(os.path.join(source_dir, '*.jpeg'))

# 随机选择4000张图片
selected_images = random.sample(image_paths, min(4000, len(image_paths)))

# 定义池化倍数列表
pooling_scales = [2, 4, 8, 16]

# 将未处理的图像复制到 ./data/GT 下
for image_path in selected_images:
    # 读取原始图像
    image = cv2.imread(image_path)
    
    # 保存原始图像到GT目录
    save_path = os.path.join(gt_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, image)

# 处理后的图像保存到 ./data/hazy 下
for image_path in selected_images:
    # 读取图片和对应的掩码
    image = cv2.imread(image_path)
    mask_path = os.path.join(mask_dir, os.path.basename(image_path).replace('.jpeg', '.jpeg'))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    for scale in pooling_scales:
        # 对图像进行池化操作
        pooled_image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv2.INTER_AREA)
        
        # 使用双线性插值恢复到原图大小
        restored_image = cv2.resize(pooled_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 将恢复后的图像与掩码相乘
        masked_image = cv2.bitwise_and(restored_image, restored_image, mask=mask)
        
        # 保存处理后的图像
        output_dir = os.path.join(hazy_dir_base, f'pooling_{scale}x')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, masked_image)
