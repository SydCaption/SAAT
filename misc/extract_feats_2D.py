import cv2
import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
import os
#from inceptionresnetv2 import inceptionresnetv2
from resnet import resnet101
import torchvision.transforms as trn
import torch
import argparse

def extract_feats(file_path, filenames, frame_num, batch_size, save_path):
	"""Extract 2D features (saved in .npy) for frames in a video."""
	#net = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background', load_path='./pretrained_models/inceptionresnetv2-520b38e4.pth')
	net = resnet101(pretrained=True)
	net.eval()
	net.cuda()
	transform = trn.Compose([trn.ToPILImage(),
		trn.Resize((224, 224)), # 299 for IRV2
		trn.ToTensor(),
		trn.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])#trn.Normalize(net.mean, net.std)])
		
	print("res101 Network loaded")
	#Read videos and extract features in batches
	for fname in filenames:
		feat_file = os.path.join(save_path, fname[:-4]+'.npy')
		if os.path.exists(feat_file):
			continue
		vid = imageio.get_reader(os.path.join(file_path, fname), 'ffmpeg')
		curr_frames = []
		for frame in vid:
			if len(frame.shape)<3:
				frame = np.repeat(frame,3)
			curr_frames.append(transform(frame).unsqueeze(0))
		curr_frames = torch.cat(curr_frames, dim=0)
		print("Shape of frames: {0}".format(curr_frames.shape))
		idx = np.linspace(0, len(curr_frames)-1, frame_num).astype(int)
		curr_frames = curr_frames[idx,:,:,:].cuda()
		print("Captured {} frames: {}".format(frame_num, curr_frames.shape))
		
		curr_feats = []
		for i in range(0, frame_num, batch_size):
			curr_batch = curr_frames[i:i+batch_size,:,:,:]
			out = net(curr_batch)
			curr_feats.append(out.detach().cpu())
			print("Appended {} features {}".format(i+1,out.shape))
		curr_feats = torch.cat(curr_feats, 0)
		del out
		np.save(feat_file,curr_feats.numpy())
		print("Saved file {}\nExiting".format(fname[:-4] + '.npy'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='./Data')
	parser.add_argument('--dataset_name', type=str, default='YouTubeClips')
	parser.add_argument('--frame_per_video', type=int, default=28)
	parser.add_argument('--start_idx', type=int, default=0)
	parser.add_argument('--end_idx', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=1)
	opt = parser.parse_args()

	save_path = os.path.join(opt.file_path, 'Feature_2D')
	namelist = os.listdir(os.path.join(opt.file_path, opt.dataset_name))
	extract_feats(opt.file_path, namelist[opt.start_idx:opt.end_idx], opt.frame_per_video, opt.batch_size, save_path)
