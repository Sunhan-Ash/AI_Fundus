import os
import shutil
import random
from glob import glob

# 定义源目录和目标目录
gt_dir = './data/eye_pooled3/GT/'
hazy_dir = './data/eye_pooled3/hazy/'
train_dir = './data/eye_pooled3/train/'
test_dir = './data/eye_pooled3/test/'

# 创建目标目录
train_gt_dir = os.path.join(train_dir, 'GT/')
train_hazy_dir = os.path.join(train_dir, 'hazy/')
test_gt_dir = os.path.join(test_dir, 'GT/')
test_hazy_dir = os.path.join(test_dir, 'hazy/')

os.makedirs(train_gt_dir, exist_ok=True)
os.makedirs(train_hazy_dir, exist_ok=True)
os.makedirs(test_gt_dir, exist_ok=True)
os.makedirs(test_hazy_dir, exist_ok=True)

# 获取GT目录中的所有图像路径
gt_images = glob(os.path.join(gt_dir, '*.jpeg'))

# 定义hazy图像的倍数后缀
scales = ['8x', '16x', '6x']

# 创建用于存储分割后图像路径的列表
train_gt_images, test_gt_images = [], []
train_hazy_images, test_hazy_images = [], []

# 随机打乱GT图像路径
random.shuffle(gt_images)

# 计算分割索引
split_index = int(len(gt_images) * 0.9)

# 分割GT图像路径
train_gt_images = gt_images[:split_index]
test_gt_images = gt_images[split_index:]

# 根据GT图像路径生成对应的hazy图像路径
for gt_image in train_gt_images:
    base_name = os.path.splitext(os.path.basename(gt_image))[0]
    for scale in scales:
        hazy_image = os.path.join(hazy_dir, f'{base_name}_{scale}.jpeg')
        if os.path.exists(hazy_image):
            train_hazy_images.append(hazy_image)

for gt_image in test_gt_images:
    base_name = os.path.splitext(os.path.basename(gt_image))[0]
    for scale in scales:
        hazy_image = os.path.join(hazy_dir, f'{base_name}_{scale}.jpeg')
        if os.path.exists(hazy_image):
            test_hazy_images.append(hazy_image)

# 将训练集的GT图像复制到 train/GT 目录中
for img_path in train_gt_images:
    shutil.copy(img_path, os.path.join(train_gt_dir, os.path.basename(img_path)))

# 将测试集的GT图像复制到 test/GT 目录中
for img_path in test_gt_images:
    shutil.copy(img_path, os.path.join(test_gt_dir, os.path.basename(img_path)))

# 将训练集的hazy图像复制到 train/hazy 目录中
for img_path in train_hazy_images:
    shutil.copy(img_path, os.path.join(train_hazy_dir, os.path.basename(img_path)))

# 将测试集的hazy图像复制到 test/hazy 目录中
for img_path in test_hazy_images:
    shutil.copy(img_path, os.path.join(test_hazy_dir, os.path.basename(img_path)))

print("数据集分割完成，图像已保存到相应的目录中。")
