import os
import shutil

# 定义包含图片的源目录和目标基目录
source_dir = 'train'  # 假设 'train' 是包含图片的文件夹
destination_base_dir = './dataset/'

# 为每个质量标签创建新的文件夹
quality_labels = df['quality'].unique()  # 获取所有唯一的质量标签
destination_folders = {label: os.path.join(destination_base_dir, f'quality_{label}') for label in quality_labels}

# 如果目录不存在，则创建目录
for folder in destination_folders.values():
    os.makedirs(folder, exist_ok=True)

# 将图片复制到相应的质量文件夹中
for index, row in df.iterrows():
    image_name = row['image']  # 图片名称
    quality_label = row['quality']  # 图片的质量标签
    
    source_path = os.path.join(source_dir, image_name)  # 原始图片路径
    destination_path = os.path.join(destination_folders[quality_label], image_name)  # 目标路径
    
    if os.path.exists(source_path):  # 确保源图片存在
        shutil.copy2(source_path, destination_path)  # 复制图片到目标路径

destination_folders
