
#### 有问题，别用
import cv2
import numpy as np
import os
from glob import glob

# 定义加载图像的目录和保存掩码的目录
source_dirs = ['./dataset/diabetic-retinopathy-detection/quality_0']
mask_base_dir = './dataset/diabetic-retinopathy-detection/mask'

# 创建保存掩码的目录
os.makedirs(mask_base_dir, exist_ok=True)

# 处理每个目录中的图像
for source_dir in source_dirs:
    # 获取所有图像文件的路径
    image_paths = glob(os.path.join(source_dir, '*.jpeg'))
    
    for image_path in image_paths:
        # 读取图像并转换为灰度图像
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建一个与灰度图像大小相同的掩码，并将所有非零像素设置为1（感兴趣区域）
        mask = np.ones_like(gray)
        
        # 将灰度图像中值为0的区域设置为0（表示边界）
        mask[gray == 0] = 0
        
        # 将掩码转换为二值图像格式（0和255）
        binary_mask = mask * 255
        
        # 保存掩码图像
        mask_dir = os.path.join(mask_base_dir, os.path.basename(source_dir))
        os.makedirs(mask_dir, exist_ok=True)
        save_path = os.path.join(mask_dir, os.path.basename(image_path).replace('.jpeg', '.jpeg'))
        cv2.imwrite(save_path, binary_mask)  # 保存为二值图像格式


print('Done!')