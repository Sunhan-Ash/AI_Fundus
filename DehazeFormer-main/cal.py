import os
import cv2
import pyiqa

# 计算 PSNR 和 SSIM 的函数假设已经存在
# psnr_value = calculate_psnr(img1, img2)
# ssim_value = calculate_ssim(img1, img2)
calculate_psnr = pyiqa.create_metric('psnr', device = 'cuda:0')
calculate_ssim = pyiqa.create_metric('ssim', device = 'cuda:0')
def calculate_metrics_for_dataset(img_folder, target_folder):
    psnr_results = 0
    ssim_results = 0
    count = 0
    # 遍历图像文件夹中的所有文件
    for img_name in os.listdir(img_folder):
        count+=1
        # 根据 img_name 生成 target_name
        target_image_name = img_name.split('_')[0] + '.' + img_name.split('.')[-1]

        # 构造图像和目标图像的完整路径
        img_path = os.path.join(img_folder, img_name)
        target_path = os.path.join(target_folder, target_image_name)

        # 读取输入图像和目标图像
        # img = cv2.imread(img_path)
        # target_img = cv2.imread(target_path)

        # # 检查是否成功读取图像
        # if img is None or target_img is None:
        #     print(f"Error reading image or target: {img_name} or {target_image_name}")
        #     continue

        # 调用现有的 PSNR 和 SSIM 计算函数
        psnr_value = calculate_psnr(img_path, target_path)
        ssim_value = calculate_ssim(img_path, target_path)

        # 存储结果
        psnr_results+=psnr_value.item()
        ssim_results+=ssim_value.item()

    return psnr_results/count, ssim_results/count

# 示例调用
img_folder = '/media/xusunhan/ZhiTai/AI_fundus/DehazeFormer-main/data/eye_pooled2/test/hazy'
target_folder = '/media/xusunhan/ZhiTai/AI_fundus/DehazeFormer-main/data/eye_pooled2/test/GT'

psnr_results, ssim_results = calculate_metrics_for_dataset(img_folder, target_folder)
print(f"PSNR = {psnr_results}, SSIM = {ssim_results}")
# 打印结果
# for i, (psnr, ssim) in enumerate(zip(psnr_results, ssim_results)):
#     print(f"Image {i+1}: PSNR = {psnr}, SSIM = {ssim}")
