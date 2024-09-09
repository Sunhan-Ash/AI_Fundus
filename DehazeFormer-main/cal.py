import os
import cv2
import pyiqa

# 计算 PSNR 和 SSIM 的函数假设已经存在
# psnr_value = calculate_psnr(img1, img2)
# ssim_value = calculate_ssim(img1, img2)
calculate_psnr = pyiqa.create_metric('psnr', device = 'cuda:0')
calculate_ssim = pyiqa.create_metric('ssim', device = 'cuda:0')
def calculate_metrics_for_dataset(img_folder, target_folder):
    psnr_results = []
    ssim_results = []

    # 遍历图像文件夹中的所有文件
    for img_name in os.listdir(img_folder):
        # 根据 img_name 生成 target_name
        target_image_name = img_name.split('_')[0] + '.' + img_name.split('.')[-1]

        # 构造图像和目标图像的完整路径
        img_path = os.path.join(img_folder, img_name)
        target_path = os.path.join(target_folder, target_image_name)

        # 读取输入图像和目标图像
        img = cv2.imread(img_path)
        target_img = cv2.imread(target_path)

        # 检查是否成功读取图像
        if img is None or target_img is None:
            print(f"Error reading image or target: {img_name} or {target_image_name}")
            continue

        # 调用现有的 PSNR 和 SSIM 计算函数
        psnr_value = calculate_psnr(img, target_img)
        ssim_value = calculate_ssim(img, target_img)

        # 存储结果
        psnr_results.append(psnr_value)
        ssim_results.append(ssim_value)

    return psnr_results, ssim_results

# 示例调用
img_folder = '/path/to/img_folder'
target_folder = '/path/to/target_folder'

psnr_results, ssim_results = calculate_metrics_for_dataset(img_folder, target_folder)

# 打印结果
for i, (psnr, ssim) in enumerate(zip(psnr_results, ssim_results)):
    print(f"Image {i+1}: PSNR = {psnr}, SSIM = {ssim}")
