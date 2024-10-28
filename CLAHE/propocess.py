import cv2
import os
from PIL import Image, ImageEnhance
# from skimage.restoration import estimate_sigma
# from BM3D.BM3D.bm3d import bm3d_rgb, BM3DProfile
import numpy as np
# from bm3d import bm3d_rgb, BM3DProfile

def calculate_brightness(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取亮度通道（V通道）
    v_channel = hsv[:, :, 2]
    
    # 计算亮度平均值
    brightness = np.mean(v_channel)
    return brightness

def adjust_brightness(image, brightness_factor):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened_image = enhancer.enhance(brightness_factor)
    return cv2.cvtColor(np.array(brightened_image), cv2.COLOR_RGB2BGR)

def calculate_brightness_factor(current_brightness, target_brightness):
    # 防止零除的异常
    if current_brightness == 0:
        return 1.0
    return target_brightness / current_brightness

# 设置输入和输出文件夹路径
input_folder = '/media/xusunhan/ZhiTai/AI_fundus/pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/datasets/Mix_Small/testA'
output_folder = './output'
target_brightness = 100
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

# def BM3D_denoise(img,psd):
#         # Call BM3D With the default settings.
#         y_est = bm3d_rgb(img, psd) #* 255
#         y_est = np.clip(y_est, -1, 1) * 255


for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取彩色图像
        img = cv2.imread(os.path.join(input_folder, filename))
        current_brightness = calculate_brightness(img)


        brightness_factor = calculate_brightness_factor(current_brightness, target_brightness)

        adjusted_image = adjust_brightness(img, brightness_factor)
        # denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        denoised_img = cv2.bilateralFilter(adjusted_image, d=9, sigmaColor=75, sigmaSpace=75)
        # sigma_est = estimate_sigma(img, average_sigmas=True)
        # denoised_img = BM3D_denoise(img=img,psd=sigma_est)
        # 将图像转换为LAB颜色空间
        lab = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2LAB)
        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 拆分LAB图像为L, A, B通道
        l, a, b = cv2.split(lab)

        # 对L通道应用CLAHE
        laplacian = cv2.Laplacian(l, cv2.CV_64F)
        l_clahe = clahe.apply(laplacian)

        # 将处理后的L通道和原始的A, B通道合并
        lab_clahe = cv2.merge((l_clahe, a, b))

        # 将LAB图像转换回RGB
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_folder, filename), img_clahe)

print("CLAHE处理完成")
