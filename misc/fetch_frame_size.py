import os
import cv2
import h5py

file_path = './Data'
ds_name = 'msvd'
frame_path = os.path.join(file_path, 'Frames')
videos = os.listdir(frame_path)
frame_size = h5py.File(ds_name+'_fr_size.h5', 'w')

for video in videos:
	vid = video
	im = cv2.imread(os.path.join(frame_path, video, '1.jpg'))
	frame_size.create_dataset(vid, data=im.shape)

frame_size.close()
