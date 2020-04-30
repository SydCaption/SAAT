# Syntax-Aware Action Targeting for Video Captioning

Code for SAAT from "Syntax-Aware Action Targeting for Video Captioning" (Accepted to CVPR 2020). The implementation is based on ["Consensus-based Sequence Training for Video Captioning"](https://github.com/mynlp/cst_captioning).

## Dependencies

* Python 3.6
* Pytorch 1.1
* CUDA 10.0
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/plsang/cider)

(Check out the `coco-caption` and `cider` projects into your working directory)

## Data

Data can be downloaded [here](https://drive.google.com/drive/folders/1n0RITmiyb0vdInGNj4O7m661V7BrsSOO?usp=sharing) (1.3GB). This folder contains:
* input/msrvtt: annotatated captions (note that `val_videodatainfo.json` is a symbolic link to `train_videodatainfo.json`)
* output/feature: extracted features of IRv2, C3D and Category embeddings
* output/metadata: preprocessed annotations
* output/model_svo/xe: model file and generated captions on test videos, the reported result can be reproduced by the model provided in this folder (CIDEr 49.1 for XE training)

## Test

```bash
make -f SpecifiedMakefile test [options]
```
Please refer to the Makefile (and opts_svo.py file) for the set of available train/test options. For example, to reproduce the reported result
```bash
make -f Makefile_msrvtt_svo test GID=0 EXP_NAME=xe FEATS="irv2 c3d category" BFEATS="roi_feat roi_box" USE_RL=0 CST=0 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG LAMBDA=20
```

## Train

To train the model using XE loss
```bash
make -f Makefile_msrvtt_svo train GID=0 EXP_NAME=xe FEATS="irv2 c3d category" BFEATS="roi_feat roi_box" USE_RL=0 CST=0 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG MAX_EPOCH=100 LAMBDA=20
```

If you want to change the input features, modify the `FEATS` variable in above commands.

### Citation
```
@InProceedings{Zheng_2020_CVPR,
author = {Zheng, Qi and Wang, Chaoyue and Tao, Dacheng},
title = {Syntax-Aware Action Targeting for Video Captioning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Acknowledgements

* Pytorch implementation of [CST](https://github.com/mynlp/cst_captioning)
* PyTorch implementation of  [SCST](https://github.com/ruotianluo/self-critical.pytorch)
