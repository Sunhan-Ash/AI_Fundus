import cv2
import os

# 设置输入和输出文件夹路径
input_folder = '/media/xusunhan/ZhiTai/AI_fundus/DehazeFormer-main/data/eye_real/test/hazy'
output_folder = './output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取彩色图像
        img = cv2.imread(os.path.join(input_folder, filename))
        

        # denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        denoised_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        # 将图像转换为LAB颜色空间
        lab = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2LAB)

        # 拆分LAB图像为L, A, B通道
        l, a, b = cv2.split(lab)

        # 对L通道应用CLAHE
        l_clahe = clahe.apply(l)

        # 将处理后的L通道和原始的A, B通道合并
        lab_clahe = cv2.merge((l_clahe, a, b))

        # 将LAB图像转换回RGB
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_folder, filename), img_clahe)

print("CLAHE处理完成")
