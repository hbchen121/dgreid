![Python >=3.7](https://img.shields.io/badge/Python->=3.7-blue.svg)
![PyTorch >=1.1](https://img.shields.io/badge/PyTorch->=1.1-yellow.svg)

# Domain Generalization Person ReID Baseline

This repository is the code for the Strong Baseline in [Style Variable and Irrelevant Learning for Generalizable Person Re-identification](https://arxiv.org/abs/2209.05235). 

## Requirements

### Installation

```shell
git clone ...
cd dgreid/reid/evaluation_metrics/rank_cylib && make all
```

### Prepare Datasets

Download the person re-ID datasets [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [DukeMTMC-ReID](https://arxiv.org/abs/1701.07717), [MSMT17](https://arxiv.org/abs/1711.08565), and
cuhk03.
Then unzip them under the root directory like
```
/data/datasets/
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── cuhk03
    └── cuhk03_release
```

## Training

By default we utilize 4 GTX-2080TI GPUs for training. **Note that**

+ The multi-source domains are trained parallel with DP.
+ More details of configs in reid/config/default_parser.py

### Quickly Start
To train the baseline methods, run commands like:
```shell
# Base Baseline (w/o meta learning)
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/base_baseline.sh

# Strong Baseline (w/ Meta learning)
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/meta_baseline.sh
```

## Citation
If you find our work is useful for your research, please kindly cite our paper
```
@misc{chen2022style,
      title={Style Variable and Irrelevant Learning for Generalizable Person Re-identification}, 
      author={Haobo Chen and Chuyang Zhao and Kai Tu and Junru Chen and Yadong Li and Boxun Li},
      year={2022},
      eprint={2209.05235},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you have any questions, please leave an issue or contact us: hbchen121@gmail.com or cy.zhao15@gmail.com

## Acknowledgement
Our code is based on [MMT](https://github.com/yxgeee/MMT) and [IDM](https://github.com/SikaStar/IDM).
