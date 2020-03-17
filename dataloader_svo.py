from __future__ import print_function

import torch
import json
import h5py
import os
import numpy as np
import random
import time
from six.moves import cPickle

import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class DataLoader():

	"""Class to load video features and captions"""

	def __init__(self, opt):
		self.iterator = 0
		self.epoch = 0

		self.batch_size = opt.get('batch_size', 128)
		self.seq_per_img = opt.get('seq_per_img', 1)
		self.word_embedding_size = opt.get('word_embedding_size', 512)
		self.num_chunks = opt.get('num_chunks', 1)
		self.num_boxes = opt.get('num_boxes', 10)
		self.mode = opt.get('mode', 'train')
		self.cocofmt_file = opt.get('cocofmt_file', None)
		self.bcmrscores_pkl = opt.get('bcmrscores_pkl', None)

		# open the hdf5 info file
		logger.info('DataLoader loading h5 file: %s', opt['label_h5'])
		self.label_h5 = h5py.File(opt['label_h5'], 'r')

		self.vocab = [i for i in self.label_h5['vocab']]
		self.videos = [i for i in self.label_h5['videos']]

		self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}
		self.num_videos = len(self.videos)
		self.index = list(range(self.num_videos))

		# load the json file which contains additional information about the
		# dataset
		feat_h5_files = opt['feat_h5']
		logger.info('DataLoader loading h5 files: %s', feat_h5_files)
		self.feat_h5 = []
		self.feat_dims = []
		for ii, feat_h5_file in enumerate(feat_h5_files):
			self.feat_h5.append(h5py.File(feat_h5_files[ii], 'r'))
			self.feat_dims.append(self.feat_h5[ii][self.videos[0]].shape[0])

		self.num_feats = len(feat_h5_files)

		# load the h5 file which contains of regional features
		bfeat_h5_files = opt['bfeat_h5']
		logger.info('DataLoader loading bh5 files: %s', bfeat_h5_files)
		self.bfeat_h5 = []
		self.bfeat_dims = []
		for ii, bfeat_h5_file in enumerate(bfeat_h5_files):
			self.bfeat_h5.append(h5py.File(bfeat_h5_files[ii], 'r'))
			self.bfeat_dims.append(self.bfeat_h5[ii][self.videos[0]].shape[1])
		self.num_bfeats = len(bfeat_h5_files)
		
		self.fr_size = h5py.File(opt['fr_size_h5'], 'r')

		# load in the sequence data
		if 'labels' in self.label_h5.keys():
			self.seq_length = self.label_h5['labels'].shape[1]
			logger.info('max sequence length in data is: %d', self.seq_length)

			# load the pointers in full to RAM (should be small enough)
			self.label_start_ix = self.label_h5['label_start_ix']
			self.label_end_ix = self.label_h5['label_end_ix']
			assert(self.label_start_ix.shape[0] == self.label_end_ix.shape[0])
			self.has_label = True

			self.label_start_ix_svo = self.label_h5['label_start_ix_svo']
			self.label_end_ix_svo = self.label_h5['label_end_ix_svo']
			assert(self.label_start_ix_svo.shape[0] == self.label_end_ix_svo.shape[0])
			
		else:
			self.has_label = False

		if self.bcmrscores_pkl is not None:
			eval_metric = opt.get('eval_metric', 'CIDEr')
			logger.info('Loading: %s, with metric: %s', self.bcmrscores_pkl, eval_metric)
			self.bcmrscores = cPickle.load(open(self.bcmrscores_pkl, 'rb'))
			if eval_metric == 'CIDEr' and eval_metric not in self.bcmrscores:
				eval_metric = 'cider'
			self.bcmrscores = self.bcmrscores[eval_metric]
		
		if self.mode == 'train' or self.mode == 'val':
			self.shuffle_videos()

	def __del__(self):
		for f in self.feat_h5:
			f.close()
		self.label_h5.close()

	def get_batch(self):
		video_batch = []
		bb_batch = []
		for dim in self.feat_dims:
			feat = torch.FloatTensor(
				self.batch_size, self.num_chunks, dim).zero_()
			video_batch.append(feat)
		for dim in self.bfeat_dims:
			bfeat = torch.FloatTensor(
				self.batch_size, self.num_boxes, dim).zero_()
			bb_batch.append(bfeat)

		if self.has_label:
			label_batch = torch.LongTensor(
				self.batch_size * self.seq_per_img,
				self.seq_length).zero_()
			mask_batch = torch.FloatTensor(
				self.batch_size * self.seq_per_img,
				self.seq_length).zero_()
			label_svo_batch = torch.LongTensor(
				self.batch_size * self.seq_per_img,
				3).zero_()

		videoids_batch = []
		gts = []
		bcmrscores = np.zeros((self.batch_size, self.seq_per_img)) if self.bcmrscores_pkl is not None else None
		
		for ii in range(self.batch_size):
			idx = self.index[self.iterator]
			video_id = int(self.videos[idx])
			videoids_batch.append(video_id)

			for jj in range(self.num_feats):
				video_batch[jj][ii] = torch.from_numpy(
					np.array(self.feat_h5[jj][str(video_id)]))

			bb_check = []
			for jj in range(self.num_bfeats):
				cur_bfeat = np.array(self.bfeat_h5[jj][str(video_id)])
				cur_nb = cur_bfeat.shape[0]
				if cur_nb > 0:        
					bb_check.append(cur_nb)
					cur_idx = [a % cur_nb for a in range(self.num_boxes)]
					bb_batch[jj][ii] = torch.from_numpy(cur_bfeat[cur_idx,:])
				else:
					bb_check.append(0)
					bb_batch[jj][ii] = torch.rand(self.num_boxes, self.bfeat_dims[jj])
			assert min(bb_check) == max(bb_check), 'Wrong rois detected!'
			if self.has_label:
				# fetch the sequence labels
				ix1 = self.label_start_ix[idx]
				ix2 = self.label_end_ix[idx]
				ncap = int(ix2 - ix1)  # number of captions available for this image
				assert ncap > 0, 'No captions!!'

				seq = torch.LongTensor(
					self.seq_per_img, self.seq_length).zero_()
				seq_all = torch.from_numpy(
					np.array(self.label_h5['labels'][ix1:ix2]))
				if ncap <= self.seq_per_img:
					seq[:ncap] = seq_all[:ncap]
					for q in range(ncap, self.seq_per_img):
						ixl = np.random.randint(ncap)
						seq[q] = seq_all[ixl]
				else:
					randpos = torch.randperm(ncap)
					for q in range(self.seq_per_img):
						ixl = randpos[q]
						seq[q] = seq_all[ixl]

				il = ii * self.seq_per_img
				label_batch[il:il + self.seq_per_img] = seq

				# fetch the sequence svo labels
				ix1_svo = self.label_start_ix_svo[idx]
				ix2_svo = self.label_end_ix_svo[idx]
				nsvo = int(ix2_svo - ix1_svo)  # number of captions available for this image
				assert nsvo > 0, 'No svos!!'

				seq_svo = torch.LongTensor(
					self.seq_per_img, 3).zero_()
				seq_all_svo = torch.from_numpy(
					np.array(self.label_h5['labels_svo'][ix1_svo:ix2_svo]))
				if nsvo <= self.seq_per_img:
					seq_svo[:nsvo] = seq_all_svo[:nsvo]
					for q in range(nsvo, self.seq_per_img):
						ixl = np.random.randint(nsvo)
						seq_svo[q] = seq_all_svo[ixl]
				else:
					randpos = torch.randperm(nsvo)
					for q in range(self.seq_per_img):
						ixl = randpos[q]
						seq_svo[q] = seq_all_svo[ixl]

				label_svo_batch[il:il + self.seq_per_img] = seq_svo
				# Used for reward evaluation
				gts.append(
					self.label_h5['labels'][
						self.label_start_ix[idx]: self.label_end_ix[idx]])
				# pre-computed cider scores, 
				# assuming now that videos order are same (which is the sorted videos order)
				if self.bcmrscores_pkl is not None:
					bcmrscores[ii] = self.bcmrscores[idx]
					
			self.iterator += 1
			if self.iterator >= self.num_videos:
				logger.info('===> Finished loading epoch %d', self.epoch)
				self.iterator = 0
				self.epoch += 1
				if self.mode == 'train' or self.mode == 'val':
					self.shuffle_videos()
		data = {}
		data['feats'] = video_batch
		data['bfeats'] = bb_batch
		data['ids'] = videoids_batch

		if self.has_label:
			# + 1 here to count the <eos> token, because the <eos> token is set to 0
			nonzeros = np.array(
				list(map(lambda x: (x != 0).sum() + 1, label_batch)))
			for ix, row in enumerate(mask_batch):
				row[:nonzeros[ix]] = 1

			data['labels_svo'] = label_svo_batch
			data['labels'] = label_batch
			data['masks'] = mask_batch
			data['gts'] = gts
			data['bcmrscores'] = bcmrscores
		return data

	def reset(self):
		self.iterator = 0

	def get_current_index(self):
		return self.iterator

	def set_current_index(self, index):
		self.iterator = index

	def get_vocab(self):
		return self.ix_to_word

	def get_vocab_size(self):
		return len(self.vocab)

	def get_feat_dims(self):
		return self.feat_dims

	def get_bfeat_dims(self):
		return self.bfeat_dims

	def get_feat_size(self):
		return sum(self.feat_dims)

	def get_num_feats(self):
		return self.num_feats

	def get_seq_length(self):
		return self.seq_length

	def get_seq_per_img(self):
		return self.seq_per_img

	def get_num_videos(self):
		return self.num_videos

	def get_batch_size(self):
		return self.batch_size

	def get_current_epoch(self):
		return self.epoch

	def set_current_epoch(self, epoch):
		self.epoch = epoch

	def shuffle_videos(self):
		np.random.shuffle(self.index)

	def get_cocofmt_file(self):
		return self.cocofmt_file
