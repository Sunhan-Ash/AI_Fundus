import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance


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

def single_scale_retinex(img, sigma):
    # 防止log10(0)导致的错误
    img = np.maximum(img, 1)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img) - np.log10(blurred)
    return retinex

def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img)
    img = np.maximum(img, 1)
    for sigma in sigmas:
        img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img) - np.log10(img_blur)
    retinex /= len(sigmas)
    return retinex

def apply_multi_scale_retinex_to_color_image(color_img, sigmas):
    # 分离颜色通道
    b, g, r = cv2.split(color_img.astype(np.float64))
    
    # 对每个通道应用多尺度Retinex
    b_retinex = multi_scale_retinex(b, sigmas)+b
    g_retinex = multi_scale_retinex(g, sigmas)+g
    r_retinex = multi_scale_retinex(r, sigmas)+r
    
    # 合并通道
    retinex_img = cv2.merge((b_retinex, g_retinex, r_retinex))
    
    # 归一化结果到0-255范围
    retinex_img = cv2.normalize(retinex_img, None, 0, 255, cv2.NORM_MINMAX)
    retinex_img = retinex_img.clip(0, 255).astype(np.uint8)
    
    return retinex_img

def enhance_contrast(img, alpha=1.5, beta=0):
    # 增强对比度
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return enhanced_img

def main():

    # 设置输入和输出文件夹路径
    input_folder = './input'
    output_folder = './output'
    target_brightness = 100
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 读取彩色图像
                # 加载图像
            img = cv2.imread(os.path.join(input_folder, filename))
            # print(f"Processing {filename} with shape {img.shape}")
            # result = SSR(img = img, sigma = 20)
            # 应用多尺度Retinex算法
            GaussianBlurzed_img = cv2.GaussianBlur(img, (5, 5), 0)
            denoised_img = cv2.bilateralFilter(GaussianBlurzed_img, d=9, sigmaColor=75, sigmaSpace=75)
            # sigmas = [15, 80, 200]  # 调整这些sigma值以获得更好的效果
            # retinex_img = apply_multi_scale_retinex_to_color_image(img, sigmas)
            
            # 增强对比度
            enhanced_img = enhance_contrast(denoised_img)

            # 保存处理后的图像
            cv2.imwrite(os.path.join(output_folder, "denoised_"+filename), GaussianBlurzed_img)
            cv2.imwrite(os.path.join(output_folder, "Enhanced_MSR_"+filename), enhanced_img)
            cv2.imwrite(os.path.join(output_folder, "Original_"+filename), img)
    
    # # 使用Matplotlib显示图像
    # plt.figure(figsize=(150, 50))
    # plt.subplot(1, 3, 1)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.title('Retinex Enhanced Image')
    # plt.imshow(cv2.cvtColor(retinex_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    
    # plt.subplot(1, 3, 3)
    # plt.title('Contrast Enhanced Image')
    # plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    
    # plt.show()

if __name__ == "__main__":
    main()







