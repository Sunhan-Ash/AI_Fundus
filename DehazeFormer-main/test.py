import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
import pyiqa

ssim = pyiqa.create_metric('ssim',device='cuda:0')
musiq = pyiqa.create_metric("musiq", device="cuda:0")
piqe = pyiqa.create_metric("piqe", device="cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='eye_paired', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	PIQE = AverageMeter()
	MUSIQ = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()
		# mask = batch['mask'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input).clamp_(-1, 1)

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			# output = output * mask
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(output, 
							target).item()	
			# piqe_val = piqe(output).item()
			piqe_val = 0
			# musiq_val = 0
			musiq_val = musiq(output).item()			

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		PIQE.update(piqe_val)
		MUSIQ.update(musiq_val)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
			  'PIQE: {piqe.val:.04f} ({piqe.avg:.04f})\t'
			  'MUSIQ: {musiq.val:.04f} ({musiq.avg:.04f})\t'
			  .format(idx, psnr=PSNR, ssim=SSIM, piqe=PIQE, musiq=MUSIQ))

		f_result.write('%s,%.02f,%.03f,%.04f,%.04f\n'%(filename, psnr_val, ssim_val,piqe_val,musiq_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)