# 构建有监督数据集
import os
import shutil

# 原始文件夹路径
source_folder = './pytorch-CycleGAN-and-pix2pix-master/results/EyeQ_cyclegan_new/test_latest/images'

# 目标文件夹路径
input_folder = '../DehazeFormer-main/data/fake_temp/train/hazy'
gt_folder = '../DehazeFormer-main/data/fake_temp/train/GT'

# 确保目标文件夹存在
os.makedirs(input_folder, exist_ok=True)
os.makedirs(gt_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith("_fake_A.png"):
        # 移除文件名中的后缀
        new_filename = filename.replace("_fake_A", "")
        # 构造原始文件路径和目标文件路径
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(input_folder, new_filename)
        # 复制文件到目标文件夹
        shutil.copy(src_path, dst_path)
        print(f"Copied and renamed: {filename} to {dst_path}")

    elif filename.endswith("_real_B.png"):
        # 移除文件名中的后缀
        new_filename = filename.replace("_real_B", "")
        # 构造原始文件路径和目标文件路径
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(gt_folder, new_filename)
        # 复制文件到目标文件夹
        shutil.copy(src_path, dst_path)
        print(f"Copied and renamed: {filename} to {dst_path}")
