# ModernSR

### Overview
[EDSR-Pytorch] (https://github.com/thstkdgus35/EDSR-PyTorch) is a widely-used code for various image restoration tasks. However, the original repo only supports old-fashioned data parallel that is less efficient on modern GPU hardware. The goal of this repo is to extend the EDSR codebase with distributed data parallel which is officially recommended by the PyTorch Team. 

It also includes several self-attention-based models ([CSNLLN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Image_Super-Resolution_With_Cross-Scale_Non-Local_Attention_and_Exhaustive_Self-Exemplars_Mining_CVPR_2020_paper.pdf), [PANet](https://arxiv.org/pdf/2004.13824.pdf), [NLSN](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf)) I developed over the past few years.

### How to use?
This repo follows the exact workflow of the original EDSR codebase except --batch_size now refers to the local batch size on a single GPU. Please check the [original repo](https://github.com/thstkdgus35/EDSR-PyTorch) for how to run the code.

### Acknowledgement
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.

