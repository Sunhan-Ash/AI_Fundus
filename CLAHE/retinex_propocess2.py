import cv2
import os
from PIL import Image, ImageEnhance
# from skimage.restoration import estimate_sigma
# from BM3D.BM3D.bm3d import bm3d_rgb, BM3DProfile
import numpy as np
# from bm3d import bm3d_rgb, BM3DProfile
def single_scale_retinex(img, sigma):
    # 分离颜色通道
    B, G, R = cv2.split(img)
    
    def process_channel(channel):
        # 高斯模糊以模拟光照分量
        blurred = cv2.GaussianBlur(channel, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        # 将原始图像和高斯模糊图像转换为浮点数并归一化到[0,1]
        channel_f = channel.astype(np.float32) / 255.0
        blurred_f = blurred.astype(np.float32) / 255.0
        
        # 确保所有值都是正数以避免log(0)或log(negative)
        channel_f = np.clip(channel_f, 1e-6, None)
        blurred_f = np.clip(blurred_f, 1e-6, None)
        
        # 计算对数变换后的反射率
        log_R = np.log(channel_f) - np.log(blurred_f)
        
        # 归一化到[0,255]范围
        log_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
        return log_R.astype(np.uint8)

    # 对每个颜色通道应用处理函数
    B_R = process_channel(B)
    G_R = process_channel(G)
    R_R = process_channel(R)

    # 合并处理后的颜色通道
    retinex_img = cv2.merge([B_R, G_R, R_R])
    return retinex_img

def multi_scale_retinex(img, sigma_list):
    weight = 1.0 / len(sigma_list)
    B, G, R = cv2.split(img)

    def process_channel(channel, sigmas):
        retinex = np.zeros_like(channel, dtype=np.float32)
        for sigma in sigmas:
            blurred = cv2.GaussianBlur(channel, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
            channel_f = channel.astype(np.float32) / 255.0
            blurred_f = blurred.astype(np.float32) / 255.0
            
            # 确保所有值都是正数以避免log(0)或log(negative)
            channel_f = np.clip(channel_f, 1e-6, None)
            blurred_f = np.clip(blurred_f, 1e-6, None)
            
            log_R = np.log(channel_f) - np.log(blurred_f)
            retinex += weight * log_R
        
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return retinex.astype(np.uint8)

    B_R = process_channel(B, sigma_list)
    G_R = process_channel(G, sigma_list)
    R_R = process_channel(R, sigma_list)

    retinex_img = cv2.merge([B_R, G_R, R_R])
    return retinex_img

# 设置输入和输出文件夹路径
# input_folder = './input_debug'
# output_folder = './output_debug'
# target_brightness = 100
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)


# for filename in os.listdir(input_folder):
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         # 读取彩色图像
#         img = cv2.imread(os.path.join(input_folder, filename))
#         print(f"Processing {filename} with shape {img.shape}")
#         result = SSR(img = img, sigma = 80)

#         # 保存处理后的图像
#         cv2.imwrite(os.path.join(output_folder, filename), result)

# print("SSR处理完成")
if __name__ == "__main__":
    input_image_path = './input_debug/0001.jpg'  # 替换为你的图像路径
    output_folder = './output_debug'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 加载图像
    img = cv2.imread(input_image_path)

    if img is None:
        print("Error: Image not found or unable to load.")
        exit()

    # 应用SSR
    sigma = 50
    ssr_result = single_scale_retinex(img, sigma=sigma)
    print(sigma)
    cv2.imwrite(os.path.join(output_folder, 'ssr_result.jpg'), ssr_result)

    # 应用MSR
    msr_result = multi_scale_retinex(img, sigma_list=[15, 80, 250])
    cv2.imwrite(os.path.join(output_folder, 'msr_result.jpg'), msr_result)

    # 显示结果
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(ssr_result, cv2.COLOR_BGR2RGB))
    axs[1].set_title('SSR Result')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(msr_result, cv2.COLOR_BGR2RGB))
    axs[2].set_title('MSR Result')
    axs[2].axis('off')

    plt.show()