import cv2
import numpy as np
import colour
from skimage import exposure
import matplotlib.pyplot as plt
import os
# Step 1: Convert RGB Image to CIECAM02 J Component
def rgb_to_ciecam02_j(image_rgb):
    """
    Convert an RGB image to the CIECAM02 color space and extract the J component (lightness).
    """
    # Convert RGB to XYZ
    image_rgb_normalized = image_rgb / 255.0  # Normalize to [0, 1]
    image_xyz = colour.RGB_to_XYZ(image_rgb_normalized, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint,
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ)

    # Define viewing conditions for CIECAM02 conversion
    illuminant = colour.models.RGB_COLOURSPACES['sRGB'].whitepoint  # D65 illuminant (sRGB whitepoint)
    L_A = 64  # Typical luminance of the adapting field (in cd/m²)
    Y_b = 20  # Background relative luminance (in cd/m²)

    # Convert XYZ to CIECAM02 using the specified viewing conditions
    image_ciecam02 = colour.XYZ_to_CIECAM02(image_xyz, XYZ_w=illuminant, L_A=L_A, Y_b=Y_b)

    # Extract the J (lightness) component
    j_component = image_ciecam02[..., 0]  # J component represents the lightness
    return j_component

# Step 2: Non-linear Contrast Enhancement using CLAHE and Laplacian Pyramid
def enhance_contrast(j_component):
    """
    Enhance the contrast of the J component (lightness) using CLAHE for adaptive contrast enhancement,
    followed by Laplacian pyramid-based edge enhancement.
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Apply CLAHE to enhance local contrast
    j_enhanced = clahe.apply(np.uint8(j_component * 255))  # CLAHE requires uint8, so scale to [0, 255]
    j_enhanced = j_enhanced / 255.0  # Scale back to [0, 1] after applying CLAHE
    
    # Apply Laplacian Pyramid for edge enhancement
    laplacian = cv2.Laplacian(j_enhanced, cv2.CV_64F)  # Compute Laplacian for edge detection
    j_enhanced = j_enhanced + laplacian  # Sharpen by adding the Laplacian (edge information)
    j_enhanced = np.clip(j_enhanced, 0, 1)  # Ensure the values stay within [0, 1] to avoid overflow
    
    return j_enhanced

# Main Function to Enhance the Fundus Image
def enhance_fundus_image(image):
    """
    Enhance the fundus image by converting it to the CIECAM02 color space,
    extracting the J component, and applying contrast enhancement.
    """
    # Step 1: Read the input image (in RGB format)
    # image_rgb = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    
    # Step 2: Convert the RGB image to CIECAM02 color space and extract the J component
    j_component = rgb_to_ciecam02_j(image_rgb)
    
    # Step 3: Enhance the J component using contrast enhancement techniques
    j_enhanced = enhance_contrast(j_component)
    
    # Step 4: Display the original and enhanced J component side by side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original J Component")
    plt.imshow(j_component, cmap='gray')  # Show the original J component (grayscale)
    plt.subplot(1, 2, 2)
    plt.title("Enhanced J Component")
    plt.imshow(j_enhanced, cmap='gray')  # Show the enhanced J component (grayscale)
    plt.show()

    return j_enhanced


input_folder = '/media/xusunhan/ZhiTai/AI_fundus/pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/datasets/Mix_Small/testA'
output_folder = './output'
target_brightness = 100
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # 读取彩色图像
        img = cv2.imread(os.path.join(input_folder, filename))
        img_clahe = enhance_fundus_image(img)
        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_folder, filename), img_clahe)

# Example usage of the function with a sample image path
enhance_fundus_image('fundus_image.jpg')
