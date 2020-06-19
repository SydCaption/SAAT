import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os
import h5py

config_file = "../configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
)

if __name__ == '__main__':
	frame_path = './Data/Frames'
	feat_path = './Data/roi_feat_fc'
	vid_names = os.listdir(frame_path)
	
	roi_feats_h5 = h5py.File(os.path.join(feat_path, 'roi_feats.h5'), 'w')
	roi_box_h5 = h5py.File(os.path.join(feat_path, 'roi_box.h5'), 'w')

	for cnt, vid in enumerate(vid_names):
		print('{}/({}, {})'.format(cnt, opt.start_idx, opt.end_idx))
		curr_fr_path = os.path.join(frame_path, vid)
	
		for i in range(14, 15):
			img = cv2.imread(os.path.join(curr_fr_path, str(i)+'.jpg'))
			result, top_preds, top_roi_feats = coco_demo.run_on_opencv_image(img)
			if top_roi_feats.shape[0] > 0:
				roi_feats_h5.create_dataset(vid, data=top_roi_feats.numpy(), dtype='f4')
				roi_box_h5.create_dataset(vid, data=top_preds.bbox.numpy(), dtype='f4')


	print('done')
	roi_feats_h5.close()
	roi_box_h5.close()
