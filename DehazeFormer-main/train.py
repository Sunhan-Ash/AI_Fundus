import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
import pyiqa

musiq = pyiqa.create_metric("musiq", device="cuda:0")

piqe = pyiqa.create_metric("piqe", device="cuda:0")


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='eye_pooled', type=str, help='dataset name')# eye_degrade
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
parser.add_argument('--resume', default='True', type=bool, help='continue training from last checkpoint')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()
	for batch in tqdm(train_loader, desc="Training", unit="batch"):
	# for batch in train_loader:

		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output = network(source_img)
			loss = criterion(output, target_img)

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()
	MUSIQ = AverageMeter()
	PIQE = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()
	for batch in tqdm(val_loader, desc="Testing", unit="batch"):
	# for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(source_img).clamp_(-1, 1)
			output = output * 0.5 + 0.5		

		mse_loss = F.mse_loss(output, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		musiq_score = musiq(output).mean()
		piqe_score = piqe(output).mean()
		PSNR.update(psnr.item(), source_img.size(0))
		MUSIQ.update(musiq_score.item(), source_img.size(0))
		PIQE.update(piqe_score.item(), source_img.size(0))
	return PSNR.avg, MUSIQ.avg, PIQE.avg
	# return PSNR.avg, MUSIQ.avg, PIQE.avg

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)


	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            # batch_size=setting['batch_size'],
							batch_size=2,
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training, current model name: ' + args.model)
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		best_musiq = 0
		best_piqe = 1000
		for epoch in range(setting['epochs'] + 1):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()
			torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'last.pth'))
			if epoch % setting['eval_freq'] == 0:
				avg_psnr,avg_musiq,avg_piqe = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'psnr.pth'))
					
				if avg_musiq > best_musiq:
					best_musiq = avg_musiq
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'musiq.pth'))
				if avg_piqe < best_piqe:
					best_piqe = avg_piqe
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'piqe.pth'))	
						
				
				writer.add_scalar('best_psnr', best_psnr, epoch)
				print('Epoch: [{}/{}], Loss: {:.4f}, PSNR: {:.4f},MUSIQ:{:.4f}, PIQE:{:.4f}, Best PSNR: {:.4f}, Best MUSIQ: {:.4f}, Besy PIQE:{:.4f}'.format(epoch, setting['epochs'], loss, avg_psnr,avg_musiq, avg_piqe, best_psnr, best_musiq,best_piqe))

	elif not os.path.exists(os.path.join(save_dir, args.model+'.pth')) and args.resume:
		saved_model_dir = os.path.join(save_dir,args.model+'.pth')
		network.load_state_dict(single(saved_model_dir))
		print('==> Continue training, current model name: ' + args.model)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		best_musiq = 0
		best_piqe = 1000

		for epoch in range(setting['epochs'] + 1):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()
			torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'last.pth'))
			if epoch % setting['eval_freq'] == 0:
				avg_psnr,avg_musiq,avg_piqe = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'psnr.pth'))
					
				if avg_musiq > best_musiq:
					best_musiq = avg_musiq
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'musiq.pth'))
				if avg_piqe < best_piqe:
					best_piqe = avg_piqe
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'piqe.pth'))	
						
				
				writer.add_scalar('best_psnr', best_psnr, epoch)
				print('Epoch: [{}/{}], Loss: {:.4f}, PSNR: {:.4f},MUSIQ:{:.4f}, PIQE:{:.4f}, Best PSNR: {:.4f}, Best MUSIQ: {:.4f}, Besy PIQE:{:.4f}'.format(epoch, setting['epochs'], loss, avg_psnr,avg_musiq, avg_piqe, best_psnr, best_musiq,best_piqe))
	else:	
		print('==> Existing trained model')
		exit(1)
