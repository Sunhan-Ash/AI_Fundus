import os
import shutil

# 源文件夹路径
source_dir = "./pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/results/EyeQ_cyclegan_new/test_latest/images"
# 目标文件夹路径
target_dir = "./results/9.21"

# 图片格式列表，可以根据需要添加其他格式
image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_dir):
    # 使用os.path.splitext将文件名和扩展名分开
    name, ext = os.path.splitext(filename)
    # 检查文件是否以fake_B或real_A结尾，并且扩展名在支持的格式中
    if (name.endswith("fake_B") or name.endswith("real_A")) and any(filename.endswith(ext) for ext in image_formats):
        # 构造完整的源文件路径和目标文件路径
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        # 复制文件到目标文件夹
        shutil.copy(source_file, target_file)
        print(f"已复制: {filename}")

print("文件复制完成。")
