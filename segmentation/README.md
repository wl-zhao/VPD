# Semantic Segmentation with VPD
## Getting Started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install openmim
mim install mmcv-full
mim install mmsegmentation
```

2. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Results and Fine-tuned Models

| Model | Config | Head | Crop Size | Lr Schd | mIoU | mIoU (ms+flip)  | Fine-tuned Model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ```VPDSeg_SD-1-5``` | [config](configs/fpn_vpd_sd1-5_512x512_gpu8x2.py) | Semantic FPN | 512x512 | 80K | 53.7 | 54.6 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/78ca31e53c5549779abd/?dl=1) |


## Evaluation
Command format:
```
bash dist_train.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```
To evaluate a model with multi-scale and flip, run
```
bash dist_train.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```