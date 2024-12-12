import cv2
import numpy as np
import os

def pixel_color_amplification(image, saturation_factor=1.5):
    # 将图像从BGR色彩空间转换到Lab色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 拆分Lab图像为L, a, b通道
    l, a, b = cv2.split(lab)

    # 将a, b通道转换为浮点类型以便进行数学运算
    a = a.astype('float32')
    b = b.astype('float32')

    # 计算原始饱和度
    original_saturation = np.sqrt(a**2 + b**2)

    # 放大饱和度
    amplified_saturation = np.clip(original_saturation * saturation_factor, 0, 127)

    # 保持原来的色相不变，只放大饱和度
    ratio = np.where(original_saturation == 0, 0, amplified_saturation / original_saturation)
    a_amplified = np.clip(a * ratio, -127, 127)
    b_amplified = np.clip(b * ratio, -127, 127)

    # 合并L, a_amplified, b_amplified通道
    lab_amplified = cv2.merge((l, a_amplified.astype('uint8'), b_amplified.astype('uint8')))

    # 将Lab色彩空间转换回BGR色彩空间
    result_image = cv2.cvtColor(lab_amplified, cv2.COLOR_Lab2BGR)

    return result_image

# 示例用法
if __name__ == "__main__":
    # 读取眼底图像



    
    input_folder = './input_debug'
    output_folder = './output_debug'
    target_brightness = 100
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)


    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取彩色图像
            img = cv2.imread(os.path.join(input_folder, filename))
            print(f"Processing {filename} with shape {img.shape}")
            # 应用像素彩色放大
            amplified_image = pixel_color_amplification(img, saturation_factor=1.5)
            # result = SSR(img = img, sigma = 20)

            # 保存处理后的图像
            cv2.imwrite(os.path.join(output_folder, 'amplified_fundus_image.jpg'), amplified_image)