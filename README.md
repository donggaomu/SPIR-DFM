
<div align="center">
<h1> Self-Supervised Domain Feature Mining for Underwater Domain
Generalization Object Detection </h1>


</div>


## 👀Introduction

This repository contains the code for our paper `Self-Supervised Domain Feature Mining for Underwater Domain
Generalization Object Detection`.

![](figs/fig1.png)

## 💡Environment

We test our codebase with `PyTorch 1.10.0 + CUDA 11.3 + MMdetection2.25.1`. Please install corresponding PyTorch and CUDA versions according to your computational resources.


## ⏳Setup

### Datasets
The S-UODAC2020 dataset can be downloaded from -[This link](https://github.com/mousecpn/DMC-Domain-Generalization-for-Underwater-Object-Detection)




### Training

    $ python tools/train.py configs/spir_faster.py



### Evaluation
Run the evaluation by:

    $ python tools/test.py configs/spir_faster.py <path/to/checkpoints>

We will provide our Training weight at this link:


## 🙏Acknowledgements

Ou codes are based on [mmdetection](https://github.com/open-mmlab/mmdetection).
We also appreciate DMC(https://github.com/mousecpn/DMC-Domain-Generalization-for-Underwater-Object-Detection) for providing their processed datasets.




