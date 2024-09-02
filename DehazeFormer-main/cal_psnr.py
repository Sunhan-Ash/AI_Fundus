import os
import cv2
import numpy as np
import pyiqa
from utils import AverageMeter
import torch

calculate_psnr = pyiqa.create_metric('psnr', device = 'cuda:0')
calculate_MUSIQ = pyiqa.create_metric('musiq', device = 'cuda:0')
calculate_PIQE = pyiqa.create_metric('piqe', device = 'cuda:0')
# calculate_FID = pyiqa.create_metric('fid', device = 'cuda:0')

def get_image_pairs(dir1, dir2):
    """Get list of common image file paths in two directories."""
    images1 = set(os.listdir(dir1))
    images2 = set(os.listdir(dir2))
    common_files = list(images1 & images2)
    return [(os.path.join(dir1, file), os.path.join(dir2, file)) for file in common_files]

def calculate_psnr_for_directories(dir1, dir2):
    """Calculate PSNR for images in two directories with the same filenames."""
    image_pairs = get_image_pairs(dir1, dir2)
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    
    for file1, file2 in image_pairs:
        # image1 = cv2.imread(file1)
        # image2 = cv2.imread(file2)
        
        # if image1 is not None and image2 is not None:
        #     psnr_value = calculate_psnr(image1, image2)
        #     psnr_results[os.path.basename(file1)] = psnr_value
        # else:
        #     print(f"Could not read images: {file1} or {file2}")
        
        psnr_value = calculate_psnr(file1, file2)
        PSNR.update(psnr_value.item())
        
    return PSNR.avg

def calculate_musiq_for_directories(dir1, dir2):
    """Calculate PSNR for images in two directories with the same filenames."""
    image_pairs = get_image_pairs(dir1, dir2)
    MUSIQ = AverageMeter()
    PIQE =AverageMeter()
    torch.cuda.empty_cache()
    
    for file1, file2 in image_pairs:
        # image1 = cv2.imread(file1)
        # image2 = cv2.imread(file2)
        
        # if image1 is not None and image2 is not None:
        #     psnr_value = calculate_psnr(image1, image2)
        #     psnr_results[os.path.basename(file1)] = psnr_value
        # else:
        #     print(f"Could not read images: {file1} or {file2}")
        
        musiq_value = calculate_MUSIQ(file1)
        # piqe_value = calculate_PIQE(file1)
        MUSIQ.update(musiq_value.item())
        # PIQE.update(piqe_value.item())
        print(
			  'PIQE: {piqe.val:.04f} ({piqe.avg:.04f})\t'
			  'MUSIQ: {musiq.val:.04f} ({musiq.avg:.04f})\t'
			  .format(piqe=PIQE, musiq=MUSIQ))

        
    return MUSIQ.avg, PIQE.avg

# Example usage
dir1 = '/media/xusunhan/ZhiTai/AI_fundus/DehazeFormer-main/data/eye_degrade_last/test2/hazy'
dir2 = '/media/xusunhan/ZhiTai/AI_fundus/DehazeFormer-main/data/eye_degrade_last/test2/hazy'
# psnr_results = calculate_psnr_for_directories(dir1, dir2)
# print(psnr_results)
musiq_results = calculate_musiq_for_directories(dir1, dir2)
print(musiq_results)
# Print PSNR results
# for filename, psnr in psnr_results.items():
#     print(f"{filename}: PSNR = {psnr:.2f} dB")
