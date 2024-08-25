import os
import cv2
import numpy as np
import pyiqa
from utils import AverageMeter
import torch

calculate_psnr = pyiqa.create_metric('psnr', device = 'cuda:0')

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
        image1 = cv2.imread(file1)
        # image2 = cv2.imread(file2)
        
        # if image1 is not None and image2 is not None:
        #     psnr_value = calculate_psnr(image1, image2)
        #     psnr_results[os.path.basename(file1)] = psnr_value
        # else:
        #     print(f"Could not read images: {file1} or {file2}")
        
        psnr_value = calculate_psnr(file1, file2)
        PSNR.update(psnr_value.item())
        
    return PSNR.avg

# Example usage
dir1 = './data/fake_temp/train/hazy'
dir2 = './data/fake_temp/train/GT'
psnr_results = calculate_psnr_for_directories(dir1, dir2)
print(psnr_results)
# Print PSNR results
# for filename, psnr in psnr_results.items():
#     print(f"{filename}: PSNR = {psnr:.2f} dB")
