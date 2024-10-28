import cv2
import numpy as np
import colour
from skimage import exposure
import matplotlib.pyplot as plt
import os
def rgb_to_ciecam02(image_rgb):
    """
    Convert an RGB image to the CIECAM02 color space and extract the J, C, h components.
    """
    # Convert RGB to XYZ
    image_rgb_normalized = image_rgb   # Normalize to [0, 1]
    image_xyz = colour.RGB_to_XYZ(image_rgb_normalized, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint,
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ)

    # Define the XYZ tristimulus values for the D65 illuminant (the white point in XYZ)
    illuminant_XYZ_w = [95.047, 100.000, 108.883]  # D65 white point

    # Define other viewing conditions for CIECAM02 conversion
    L_A = 318.31  # Increased luminance of the adapting field (in cd/m²)
    Y_b = 20  # Background relative luminance (in cd/m²)

    # Convert XYZ to CIECAM02 using the D65 white point and specified viewing conditions
    image_ciecam02 = colour.XYZ_to_CIECAM02(image_xyz, XYZ_w=illuminant_XYZ_w, L_A=L_A, Y_b=Y_b)

    # Extract J (lightness), C (chroma), and h (hue) components
    j_component = image_ciecam02.J
    c_component = image_ciecam02.C
    h_component = image_ciecam02.h

    # Debugging print statements to check values
    print("J component (Lightness) range:", np.min(j_component), np.max(j_component))
    print("C component (Chroma) range:", np.min(c_component), np.max(c_component))
    print("h component (Hue) range:", np.min(h_component), np.max(h_component))

    # Normalize J component to [0, 100] range if necessary
    j_component = np.clip(j_component, 0, 100)

    return j_component, c_component, h_component

# Step 2: Reconstruct Image using Enhanced J and Original Chroma and Hue
def ciecam02_to_rgb(j_component, c_component, h_component, illuminant_XYZ_w, L_A=318.31, Y_b=20):
    """
    Reconstruct an RGB image from the enhanced J component and original C and h components.
    """
    # Reconstruct CIECAM02 object with the enhanced J component and original C and h
    image_ciecam02 = colour.CAM_Specification_CIECAM02(J=j_component, C=c_component, h=h_component)

    # Convert back to XYZ
    image_xyz = colour.CIECAM02_to_XYZ(image_ciecam02, XYZ_w=illuminant_XYZ_w, L_A=L_A, Y_b=Y_b)

    # Debugging print statements to check values
    # print("XYZ values range after conversion from CIECAM02:", np.min(image_xyz), np.max(image_xyz))

    # Clip XYZ to avoid abnormal values
    # image_xyz = np.clip(image_xyz, 0, 100)

    # Convert XYZ back to RGB
    image_rgb = colour.XYZ_to_RGB(image_xyz, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint,
                                  colour.models.RGB_COLOURSPACES['sRGB'].whitepoint, 
                                  colour.models.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

    # # Ensure the RGB values are within [0, 1]
    # image_rgb = np.clip(image_rgb, 0, 1)
    image_rgb = image_rgb*255
    # # Debugging print statements to check RGB values
    # print("RGB values range before scaling:", np.min(image_rgb), np.max(image_rgb))

    # Scale back to [0, 255] and convert to uint8 for display
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    print("RGB values range before scaling:", np.min(image_rgb), np.max(image_rgb))
    return image_rgb

# Enhance J component
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


input_folder = './CLAHE/input'
# input_folder = './input'
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
