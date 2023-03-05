# VPD
Created by [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Yongming Rao](https://raoyongming.github.io/)\*,  [Zuyan Liu](https://scholar.google.com/citations?user=7npgHqAAAAAJ&hl=en)\*, [Benlin Liu](https://liubl1217.github.io), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)†

This repository contains PyTorch implementation for paper "Unleashing Text-to-Image Diffusion Models for Visual Perception". 

VPD (<ins>**V**</ins>isual <ins>**P**</ins>erception with Pre-trained <ins>**D**</ins>iffusion Models) is a framework that leverages the high-level and low-level knowledge of a pre-trained text-to-image diffusion model to downstream visual perception tasks.

![intro](figs/intro.png)

[[Project Page]](https://vpd.ivg-research.xyz) [[arXiv]](https://arxiv.org/abs/xxxx.xxxxx)


## Installation
Clone this repo, and run
```
git submodule init
git submodule update
```
Download the checkpoint of stable-diffusion (we use `v1-5` by default) and put it in the `checkpoints` folder

## Semantic Segmentation with VPD
Equipped with a lightweight Semantic FPN and trained for 80K iterations on $512\times512$ crops, our VPD can achieve 54.7 mIoU on ADE20K.

Please check [segmentation.md](./segmentation/README.md) for detailed instructions.

## Referring Image Segmentation with VPD
VPD achieves xx.xx, xx.xx, and xx.xx oIoU on the validation sets of RefCOCO, RefCOCO+, and G-Ref, repectively.

Please check [refer.md](./refer/README.md) for detailed instructions.

## Depth Estimation with VPD
VPD obtains 0.254 RMSE on NYUv2 depth estimation benchmark, establishing the new state-of-the-art.

|  | RMSE | d1 | d2 | d3 | REL  | log_10 |
|-------------------|-------|-------|--------|--------|--------|-------|
| **VPD** | 0.254 | 0.964 | 0.995 | 0.999 | 0.069 | 0.030 |

Please check [depth.md](./depth/README.md) for detailed instructions.

## License
MIT License

## Acknowledgements

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhao2023unleashing,
  title={Unleashing Text-to-Image Diffusion Models for Visual Perception},
  author={Zhao, Wenliang and Rao, Yongming and Liu, Zuyan and Liu, Benlin and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```