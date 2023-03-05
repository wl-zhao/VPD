# Semantic Segmentation with VPD
## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install openmim
mim
mim install mmcv-full==1.6.2
mim install mmsegmentation==0.30.0
```

2. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Results and Fine-tuned Models

| Model | Config | Head | Crop Size | Lr Schd | mIoU | mIoU (ms+flip)  | Fine-tuned Model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ```VPDSeg_SD-1-5``` | [config](configs/fpn_vpd_sd1-5_512x512_gpu8x2.py) | Semantic FPN | 512x512 | 80K | 53.6 | 54.7 | [Tsinghua Cloud]() |


## Evaluation
Command format:
```
bash dist_train.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```
To evaluate a model with multi-scale and flip, run
```
bash dist_train.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```