import os

# Directory where images are stored
source_dir = './pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/datasets/EyeQ/testA/'  # 更新为你图片文件夹的路径

# 获取目录中所有文件的列表
files = os.listdir(source_dir)

# 过滤出所有JPEG文件
jpeg_files = [file for file in files if file.lower().endswith('.jpeg')]

# 遍历并重命名文件
for i, filename in enumerate(jpeg_files, start=1):
    new_name = f"{i}_A.jpeg"
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(source_dir, new_name)
    
    # 重命名文件
    os.rename(source_path, destination_path)
    print(f"Renamed: {filename} to {new_name}")

print("All images have been renamed.")
