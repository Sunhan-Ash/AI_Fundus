import cv2
import os
from PIL import Image, ImageEnhance
# from skimage.restoration import estimate_sigma
# from BM3D.BM3D.bm3d import bm3d_rgb, BM3DProfile
import numpy as np
# from bm3d import bm3d_rgb, BM3DProfile

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])  # data中不为0数字的位置中的最小值
    data[data == 0] = min_nonzero  # data中为0的位置换为最小值
    return data
def SSR(img, sigma):
    B, G, R = cv2.split(img)

    def channel(C):
        L_C = cv2.GaussianBlur(C, (5, 5), sigma)
        C = replaceZeroes(C).astype(np.float32) / 255
        L_C = replaceZeroes(L_C).astype(np.float32) / 255
        log_R_C = cv2.subtract(cv2.log(C), cv2.log(L_C))
        
        # 使用numpy的vectorized操作进行量化
        log_R_C = cv2.normalize(log_R_C, None, 0, 255, cv2.NORM_MINMAX)
        C_uint8 = cv2.convertScaleAbs(log_R_C)
        return C_uint8

    B_uint8 = channel(B)
    G_uint8 = channel(G)
    R_uint8 = channel(R)

    image = cv2.merge((B_uint8, G_uint8, R_uint8))
    return image
# def SSR(img, sigma):
#     B, G, R = cv2.split(img)
#     def channel(C):
#         L_C = cv2.GaussianBlur(C, (5, 5), sigma)  # L(x,y)=I(x,y)*G(x,y)
#         h, w = C.shape[:2]
#         C = replaceZeroes(C)
#         C = C.astype(np.float32) / 255
#         L_C = replaceZeroes(L_C)
#         L_C = L_C.astype(np.float32) / 255
#         dst_C = cv2.log(C)  # logI(x,y)
#         dst_L_C = cv2.log(L_C)  # logL(x,y)
#         log_R_C = cv2.subtract(dst_C, dst_L_C)  # logR(x,y)=logI(x,y)−logL(x,y)
#         minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R_C)  # 量化处理
#         for i in range(h):
#             for j in range(w):
#                 log_R_C[i, j] = (log_R_C[i, j] - minvalue) * 255.0 / (maxvalue - minvalue)  # R(x,y)=(value-min)(255-0)/(max-min)
#         C_uint8 = cv2.convertScaleAbs(log_R_C)
#         return C_uint8
#     B_uint8 = channel(B)
#     G_uint8 = channel(G)
#     R_uint8 = channel(R)

#     image = cv2.merge((B_uint8, G_uint8, R_uint8))
#     return image

def MSR(img, sigma_list):
    B, G, R = cv2.split(img)
    weight = 1 / 3.0
    scales_size = 3


    def channel(C, sigma_list):
        for i in range(0, scales_size):
            C = replaceZeroes(C)
            C = C.astype(np.float32) / 255
            L_C = cv2.GaussianBlur(C, (5, 5), sigma_list[i])##L(x,y)=I(x,y)∗G(x,y)
            print(sigma_list[i])
            h, w = C.shape[:2]
            log_R_C = np.zeros((h, w), dtype=np.float32)
            L_C = replaceZeroes(L_C)
            L_C = L_C.astype(np.float32) / 255
            log_C = cv2.log(C)##logI(x,y)
            log_L_C = cv2.log(L_C)##logL(x,y)
            log_R_C += weight * cv2.subtract(log_C, log_L_C)##=logR(x,y)=w(logI(x,y)−logL(x,y))

        minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R_C)
        for i in range(h):
            for j in range(w):
                log_R_C[i, j] = (log_R_C[i, j] - minvalue) * 255.0 / (maxvalue - minvalue)  ##R(x,y)=(value-min)(255-0)/(max-min)

        C_uint8 = cv2.convertScaleAbs(log_R_C)
        return C_uint8

    B_uint8 = channel(B, sigma_list)
    G_uint8 = channel(G, sigma_list)
    R_uint8 = channel(R, sigma_list)

    image = cv2.merge((B_uint8, G_uint8, R_uint8))
    return image


def calculate_brightness(image):
    
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
        result = SSR(img = img, sigma = 20)

        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_folder, filename), result)

print("SSR处理完成")
