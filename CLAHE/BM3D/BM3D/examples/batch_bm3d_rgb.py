"""
RGB image BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
For color images, block matching is performed on the luminance channel.
"""
import sys
sys.path.append('/home/liwenjuan/wangjuan/project/BM3D/')

import numpy as np
from bm3d import bm3d_rgb, BM3DProfile
from examples.experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
from PIL import Image
import matplotlib.pyplot as plt
import os


def main():

    img_root = './datasets/places2'
    images = os.listdir(img_root)
    save_root = './places2_denoised/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    c = 1
    for im in images:

        imagename = os.path.join(img_root, im)

        # Load noise-free image
        y = np.array(Image.open(imagename)) / 255

        # Possible noise types to be generated 'gw', 'g1', 'g2', 'g3', 'g4', 'g1w',
        # 'g2w', 'g3w', 'g4w'.
        noise_type = 'g1w'
        noise_var = 0.1  # Noise variance
        seed = 0  # seed for pseudorandom noise realization
        BM3DProfile.filter_strength = 0.1


        # Generate noise with given PSD
        noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)

        z = np.atleast_3d(y) + np.atleast_3d(noise)

        # Call BM3D With the default settings.
        y_est = bm3d_rgb(z, psd) #* 255
        y_est = np.clip(y_est, -1, 1) * 255

        y_est = Image.fromarray(y_est.astype(np.uint8))

        im_name = im.replace('.png', '.jpg')
        save_name = os.path.join(save_root, im_name)
        y_est.save(save_name)

        c += 1
        #if c>3:
        #    break





if __name__ == '__main__':
    main()
