import os
import csv

# Directory where images are stored
# source_dir = './dataset/diabetic-retinopathy-detection/quality_0'  # 更新为你图片文件夹的路径
source_dir = './PCENet-Image-Enhancement-master/PCENet-Image-Enhancement-master/datasets/fiq_dataset/source_gt'
# 创建一个CSV文件来记录文件名更改
csv_file_path = os.path.join(source_dir, 'renamed_files.csv')

# 获取目录中所有文件的列表
files = os.listdir(source_dir)

# 过滤出所有JPEG文件
jpeg_files = [file for file in files if file.lower().endswith('.jpg')]

# 准备写入CSV文件
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Original Name', 'New Name'])  # 写入表头

    # 遍历并重命名文件
    for i, filename in enumerate(jpeg_files, start=1):
        new_name = f"{i}.jpg"
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(source_dir, new_name)

        # 重命名文件
        os.rename(source_path, destination_path)
        # 将更改写入CSV
        writer.writerow([filename, new_name])
        print(f"Renamed: {filename} to {new_name}")

print("All images have been renamed.")
