# Referring Image Segmentation
## Getting Started 

1. Install the required packages.

```
pip install -r requirements.txt
```

2. Prepare RefCOCO datasets following [LAVT](https://github.com/yz93/LAVT-RIS).

* Download COCO 2014 Train Images [83K/13GB] from [COCO](https://cocodataset.org/#download), and extract `train2014.zip` to `./refer/data/images/mscoco/images`

* Follow the instructions in `./refer` to download and extract `refclef.zip, refcoco.zip, refcoco+.zip, refcocog.zip` to `./refer/data`

Your dataset directory should be:

```
refer/
├──data/
│  ├── images/mscoco/images/
│  ├── refclef
│  ├── refcoco
│  ├── refcoco+
│  ├── refcocog
├──evaluation/
├──...
```

## Results and Fine-tuned Models of VPD

| Dataset | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | OIoU | Mean IoU | Weights
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
RefCOCO | 85.52 | 83.02 | 78.45 | 68.53 | 36.31 | **73.46** | 75.67 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/78c884b131ec4d9d9fe5/?dl=1)
RefCOCO+ | 76.69 | 73.93 | 69.68 | 60.98 | 32.52 | **63.93** | 67.98 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/5896714e834b4451801c/?dl=1)
RefCOCOg | 75.16 | 71.16 | 65.60 | 55.04 | 29.41 | **63.12** | 66.42 | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/55e5fc86d7b94e71a102/?dl=1)


## Training

We recommend to train VPD-RIS on 8x32G NVIDIA V100 GPUs with a total batch size of 32. We count the max length of referring sentences and set the token length of lenguage model accrodingly. The checkpoint of the best epoch would be saved at `./checkpoints/`.

* Train on RefCOCO

```
bash train.sh refcoco /path/to/logdir <NUM_GPUS> --token_length 40
```

* Train on RefCOCO+

```
bash train.sh refcoco+ /path/to/logdir <NUM_GPUS> --token_length 40
```

* Train on RefCOCOg

```
bash train.sh refcocog /path/to/logdir <NUM_GPUS> --token_length 77 --splitBy umd
```

## Evaluation

* Evaluate on RefCOCO

```
bash test.sh refcoco /path/to/vpd_ris_refcoco.pth --token_length 40
```

* Evaluate on RefCOCO+

```
bash test.sh refcoco+ /path/to/vpd_ris_refcoco+.pth --token_length 40
```

* Evaluate on RefCOCOg

```
bash test.sh refcocog /path/to/vpd_ris_gref.pth --token_length 77 --splitBy umd
```

