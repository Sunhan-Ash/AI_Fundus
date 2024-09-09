import cv2
import os

# 设置输入和输出文件夹路径
input_folder = 'path/to/your/input/folder'
output_folder = 'path/to/your/output/folder'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 处理文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取图像
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

        # 应用CLAHE
        clahe_img = clahe.apply(img)

        # 保存结果
        cv2.imwrite(os.path.join(output_folder, filename), clahe_img)

print("CLAHE处理完成")
