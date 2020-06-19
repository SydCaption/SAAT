import cv2
import imageio
import numpy as np
import os
from model import generate_model
import torchvision.transforms as trn
import torch
import argparse
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, ToTensor)

def extract_feats(file_path, net, filenames, frame_num, batch_size, save_path):
	"""Extract 3D features (saved in .npy) for a video. """
	net.eval()
	mean = get_mean(255, dataset='kinetics')
	std = get_std(255)
	transform = Compose([trn.ToPILImage(),
		Scale(112),
		CornerCrop(112, 'c'),
		ToTensor(),
		Normalize(mean, std)])
		
	print("Network loaded")
	#Read videos and extract features in batches
	for file in filenames[start_idx:end_idx]:
		feat_file = os.path.join(save_path, file[:-4] + '.npy')
		if os.path.exists(feat_file):
			continue
		vid = imageio.get_reader(os.path.join(file_path, file), 'ffmpeg')
		
		curr_frames = []
		for frame in vid:
			if len(frame.shape)<3:
				frame = np.repeat(frame,3)
			curr_frames.append(transform(frame).unsqueeze(0))
		curr_frames = torch.cat(curr_frames, dim=0)
		print("Shape of frames: {0}".format(curr_frames.shape))
		idx = np.linspace(0, len(curr_frames)-1, frame_num).astype(int)
		print("Captured {} clips: {}".format(len(idx), curr_frames.shape))
		
		curr_feats = []
		for i in range(0, len(idx), batch_size):
			curr_batch = [curr_frames[x-8:x+8,...].unsqueeze(0) for x in idx[i:i+batch_size]]
			curr_batch = torch.cat(curr_batch, dim=0).cuda()
			out = net(curr_batch.transpose(1,2).cuda())
			curr_feats.append(out.detach().cpu())
			print("Appended {} features {}".format(i+1,out.shape))
		curr_feats = torch.cat(curr_feats, 0)
		del out
		#set_trace()	
		np.save(feat_file,curr_feats.numpy())
		print("Saved file {}\nExiting".format(file[:-4] + '.npy'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='resnext')
	parser.add_argument('--model_depth', type=int, default=101)
	parser.add_argument('--pretrain_path', type=str, default='./checkpoints/resnext-101-kinetics.pth')
	parser.add_argument('--n_classes', type=int, default=400)
	parser.add_argument('--n_finetune_classes', type=int, default=400)
	parser.add_argument('--ft_begin_index', type=int, default=0)
	parser.add_argument('--resnet_shortcut', type=str, default='B')
	parser.add_argument('--resnext_cardinality', type=int, default=32)
	parser.add_argument('--sample_size', type=int, default=112)
	parser.add_argument('--sample_duration', type=int, default=16)
	parser.add_argument('--no_cuda', type=bool, default=False)
	parser.add_argument('--no_train', type=bool, default=True)

	parser.add_argument('--file_path', type=str, default='./Data')
	parser.add_argument('--dataset_name', type=str, default='YouTubeClips')
	parser.add_argument('--frame_per_video', type=int, default=28)
	parser.add_argument('--start_idx', type=int, default=0)
	parser.add_argument('--end_idx', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=1)

	opt = parser.parse_args()
	opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

	model, _ = generate_model(opt)
	
	namelist = os.listdir(os.path.join(opt.file_path, opt.dataset_name))
	save_path = os.path.join(opt.file_path, 'Feature_3D')
	extract_feats(opt.file_path, model, namelist[opt.start_idx:opt.end_idx], opt.frame_per_video, opt.batch_size, save_path)
