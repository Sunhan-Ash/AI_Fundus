import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_images(images, titles, figsize=(15, 5)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 读取图像
image_path = '../moveB/moveB/24_right_0.png'
image = cv2.imread(image_path)

# 添加斑点噪声
speckle_factor = 0.1
speckle_noise = np.random.randn(*image.shape) * speckle_factor + 1
noisy_speckle = (image * speckle_noise).clip(0, 255).astype(np.uint8)
#添加乘性噪声
multiplicative_factor = 0.1
multiplicative_noise = np.random.uniform(1 - multiplicative_factor, 1 + multiplicative_factor, size=image.shape)
noisy_multiplicative = (image * multiplicative_noise).clip(0, 255).astype(np.uint8)
# 添加量化噪声
quantized_image = ((image / 32).round() * 32).astype(np.uint8)
noisy_quantization = quantized_image
cv2.imwrite("../moveB/moveB/speckle_noise_"+"24_right_0.png", noisy_speckle)
cv2.imwrite("../moveB/moveB/multiplicative_noise_"+"24_right_0.png", noisy_multiplicative)
cv2.imwrite("../moveB/moveB/noisy_quantization_"+"24_right_0.png", noisy_quantization)

# 显示图像
# display_images(
#     [image, noisy_speckle],
#     ['Original Image', 'Noisy Speckle']
# )