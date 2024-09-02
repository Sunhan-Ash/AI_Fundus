import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img, read_resize_img,read_mask


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
		
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		#这里读取的img的图片名是hazy的，命名格式是0001_001.png，所以对应的GT的文件名是0001.png
		img_name = self.img_names[idx]
		if self.mode == 'train':
			target_image_name = img_name.split('_')[0]
			target_image_name = target_image_name +'.' +img_name.split('.')[-1]
		else:
			target_image_name = img_name
		# target_image_name = img_name.split('_')[0]
		# target_image_name = target_image_name +'.' +img_name.split('.')[-1]
		# source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		# target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		# read_resize_img
		source_img = read_resize_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		mask_img = read_mask(os.path.join(self.root_dir, 'mask', target_image_name))
		# if len(mask_img.shape) == 2:
		# 	mask_img = np.expand_dims(mask_img, axis=-1)  # 扩展维度，使其形状为 (H, W, 1)
		# if self.mode == 'train':
		# 	target_img = read_resize_img(os.path.join(self.root_dir, 'GT', target_image_name)) * 2 - 1
		# if self.mode == 'valid':
		# 	target_img = read_resize_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		target_img = read_resize_img(os.path.join(self.root_dir, 'GT', target_image_name)) * 2 - 1		
		# source_img = source_img * mask_img
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
		elif self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)
		else:
			return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'mask':mask_img, 'filename': img_name}

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
