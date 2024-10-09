import cv2
import numpy as np
import os

def calculate_brightness(image):
    # 转换为HSV颜色空间并提取亮度通道
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    brightness = np.mean(v_channel)
    return brightness

def calculate_folder_brightness(folder_path):
    total_brightness = 0
    image_count = 0
    
    for filename in os.listdir(folder_path):
        # 仅处理图片文件
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            # 确保读取图像成功
            if image is not None:
                brightness = calculate_brightness(image)
                total_brightness += brightness
                image_count += 1
    
    # 计算平均亮度
    average_brightness = total_brightness / image_count if image_count > 0 else 0
    return average_brightness

# 使用示例
folder_path = "path_to_your_folder"  # 替换为目标文件夹路径
average_brightness = calculate_folder_brightness(folder_path)
print(f"目标文件夹中图像的平均亮度为: {average_brightness}")