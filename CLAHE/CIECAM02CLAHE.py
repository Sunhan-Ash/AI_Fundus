import cv2
import numpy as np
import colour
from skimage import exposure
import matplotlib.pyplot as plt
import os
# from colour.appearance import CIECAM02, CIECAM02_VIEWING_CONDITIONS
from colour.appearance import InductionFactors_CIECAM02, CAM_Specification_CIECAM02
from colour import XYZ_to_CIECAM02, CIECAM02_to_XYZ 

def rgb_to_ciecam02(image_rgb):
    """
    Convert an RGB image to the CIECAM02 color space and extract the J, C, h components.
    """
    image_xyz = colour.sRGB_to_XYZ(image_rgb)
    # illuminant = np.array([0.95047, 1.00000, 1.08883]) # D65白点
    # 2. 定义D65白点和观察条件
    illuminant = np.array([0.95047, 1.00000, 1.08883])  # D65的三色刺激值
    L_A = 64  # 适应场亮度
    Y_b = 20  # 背景亮度因子
    surround = InductionFactors_CIECAM02(F=1.0, c=0.69, N_c=1.0)  # 'Average'的普通观察条件
    # surround = colour.CAM_Specification_CAM02(L_A=64, Y_b=20, surround=colour.VIEWING_CONDITIONS_CAM02['Average'])
    # cam02_specification = colour.appearance.XYZ_to_CAM02(image_xyz, illuminant, surround)
    cam02_specification = XYZ_to_CIECAM02(
    XYZ=image_xyz,     # XYZ值
    XYZ_w=illuminant,  # 白点
    L_A=L_A,           # 适应场亮度
    Y_b=Y_b,           # 背景亮度因子
    surround=surround,  # 环境感应因子
    discount_illuminant=False,  # 不忽略光源影响
    compute_H=False  # 不计算Hue四象限
    )

    # viewing_conditions = CIECAM02_VIEWING_CONDITIONS['Average']  # 使用'Average'标准观察条件
    # cam02 = CIECAM02(XYZ=image_xyz, XYZ_w=illuminant, L_A=64, Y_b=20, surround=viewing_conditions)
    # JCh = colour.XYZ_to_CIECAM02(image_xyz, viewing_conditions)


    return cam02_specification.J, cam02_specification.C, cam02_specification.h

# Step 2: Reconstruct Image using Enhanced J and Original Chroma and Hue
def ciecam02_to_rgb(j_component, c_component, h_component, illuminant_XYZ_w, L_A=318.31, Y_b=20):
    # 1. 构建CIECAM02模型的规格
    specification = CAM_Specification_CIECAM02(J=j_component, C=c_component, h=h_component)

    # 2. 定义D65白点和观察条件
    illuminant = np.array([0.95047, 1.00000, 1.08883])  # D65白点的三色刺激值
    L_A = 64  # 适应场亮度
    Y_b = 20  # 背景亮度因子

    # 定义周围环境感应因子
    surround = InductionFactors_CIECAM02(F=1.0, c=0.69, N_c=1.0)  # 普通观察条件

    # 3. 调用CIECAM02_to_XYZ函数，将CIECAM02的J, C, h转换为XYZ
    xyz_image = CIECAM02_to_XYZ(
        specification=specification,
        XYZ_w=illuminant,
        L_A=L_A,
        Y_b=Y_b,
        surround=surround,
        discount_illuminant=False  # 不忽略光源影响
    )

    # 4. 将XYZ转换为sRGB
    rgb_image = colour.XYZ_to_sRGB(xyz_image)
    print("RGB values range before scaling:", np.min(rgb_image), np.max(rgb_image))
    return rgb_image

# Enhance J component
def enhance_contrast(j_component):
    """
    Enhance the contrast of the J component (lightness) using CLAHE for adaptive contrast enhancement,
    followed by Laplacian pyramid-based edge enhancement.
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # Apply CLAHE to enhance local contrast
    # print("RGB values range before scaling:", np.min(j_component), np.max(j_component))
    J_laplacian = cv2.Laplacian(j_component, cv2.CV_64F)
    J_laplacian_normalized = np.clip((J_laplacian - J_laplacian.min()) / (J_laplacian.max() - J_laplacian.min()) * 255, 0, 255).astype(np.uint8)
    J_enhanced = cv2.equalizeHist(J_laplacian_normalized)
    # j_enhanced = clahe.apply(np.uint8(laplacian))  # CLAHE requires uint8, so scale to [0, 255]
    # j_enhanced = j_enhanced  # Scale back to [0, 1] after applying CLAHE

    # Apply Laplacian Pyramid for edge enhancement
    # laplacian = cv2.Laplacian(j_enhanced, cv2.CV_64F)  # Compute Laplacian for edge detection
    j_enhanced = J_enhanced*100/255  # Sharpen by adding the Laplacian (edge information)
    # j_enhanced = np.clip(j_enhanced, 0, 255)  # Ensure the values stay within [0, 1] to avoid overflow

    return j_enhanced

def nomalize(image):
    for i in range(image.shape[2]):
        min_value = np.min(image[:,:,i])
        max_value = np.max(image[:,:,i])
        print("min_value:", min_value, "max_value:", max_value)
        # 2. 线性拉伸并归一化到 [0, 1]
        image[:,:,i] = 255 * (image[:,:,i] - min_value) / (max_value - min_value)

    # 3. 将结果转换为整数类型
    normalized_image = image.astype(np.uint8)
    return normalized_image

# Main Function to Enhance the Fundus Image and Keep Color
def enhance_fundus_image(image):
    """
    Enhance the fundus image by converting it to the CIECAM02 color space,
    extracting the J, C, h components, enhancing the J component,
    and then reconstructing the enhanced color image.
    """
    # Step 1: Read the input image (in RGB format)
    # image_rgb = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    # print("RGB values range before scaling:", np.min(image_rgb), np.max(image_rgb))
    # Step 2: Convert the RGB image to CIECAM02 color space and extract J, C, h components
    j_component, c_component, h_component = rgb_to_ciecam02(image_rgb)

    # Step 3: Enhance the J component using contrast enhancement techniques
    j_enhanced = enhance_contrast(j_component)

    # Step 4: Reconstruct the RGB image using the enhanced J component and original chroma and hue
    illuminant_XYZ_w = [95.047, 100.000, 108.883]  # D65 white point
    enhanced_rgb = ciecam02_to_rgb(j_enhanced, c_component, h_component, illuminant_XYZ_w)
    # enhanced_rgb = enhanced_rgb + 128
    # enhanced_rgb = np.clip(enhanced_rgb, 0, 255).astype(np.uint8)
    # enhanced_rgb = nomalize(enhanced_rgb)
    # enhanced_rgb = np.clip(enhanced_rgb * 255, 0, 255).astype(np.uint8)

    # Step 5: Display the original and enhanced color images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)  # Show the original RGB image
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced_rgb)  # Show the enhanced RGB image
    plt.show()

    return enhanced_rgb


# input_folder = './CLAHE/input'
input_folder = './input'
output_folder = './output'
target_brightness = 100
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取彩色图像
        img = cv2.imread(os.path.join(input_folder, filename))
        img_clahe = enhance_fundus_image(img)
        print("RGB values range before scaling:", np.min(img_clahe), np.max(img_clahe))
        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_folder, filename), img_clahe)

# Example usage of the function with a sample image path
enhance_fundus_image('fundus_image.jpg')
