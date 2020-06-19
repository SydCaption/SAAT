import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
import os
import argparse

def extract_frames(output, dirname, filenames, frame_num):
	"""Extract frames in a video. """
	#Read videos and extract features in batches
	for file_cnt, fname in enumerate(filenames):
		vid = imageio.get_reader(os.path.join(output, dirname, fname), 'ffmpeg')
		idx = np.linspace(0, len(vid)-1, frame_num).astype(int).tolist()
		cnt = 1 
		frames_dir = os.path.join(output, 'Frames', fname[:-4])
		if not os.path.exists(frames_dir):
			os.mkdir(frames_dir)
		
		for i, frame in enumerate(vid):
			if len(frame.shape)<3:
				frame = np.repeat(frame,3)
			if i in idx:
				imageio.imwrite(os.path.join(frames_dir, str(cnt)+'.jpg'), frame)
				cnt += 1
				if cnt > 28:
					break
		print('{}/{} done'.format(file_cnt, len(filenames)))
		assert len(os.listdir(frames_dir)) == frame_num, 'Wrong frame number...'

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='./Data')
	parser.add_argument('--dataset_name', type=str, default='YouTubeClips')
	parser.add_argument('--frame_per_video', type=int, default=28)
	parser.add_argument('--start_idx', type=int, default=0)
	parser.add_argument('--end_idx', type=int, default=1)
	opt = parser.parse_args()

	namelist = os.listdir(os.path.join(opt.file_path, opt.dataset_name))
	extract_frames(opt.file_path, opt.dataset_name, namelist[opt.start_idx:opt.end_idx], opt.frame_per_video)
