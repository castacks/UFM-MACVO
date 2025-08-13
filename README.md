
<div align="center">
<h1>UFM: A Simple Path towards Unified Dense Correspondence with Flow</h1>

<a href="https://arxiv.org/abs/0000.00000"><img src="https://img.shields.io/badge/arXiv-0000.00000-b31b1b" alt="arXiv"></a>
<a href="https://uniflowmatch.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/infinity1096/UFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**Carnegie Mellon University**

[Yuchen Zhang](https://infinity1096.github.io/), [Nikhil Keetha](https://nik-v9.github.io/), [Chenwei Lyu](https://www.linkedin.com/in/chenwei-lyu/), [Bhuvan Jhamb](https://www.linkedin.com/in/bhuvanjhamb/), [Yutian Chen](https://www.yutianchen.blog/about/)
[Yuheng Qiu](https://haleqiu.github.io), [Jay Karhade](https://jaykarhade.github.io/), [Shreyas Jha](https://www.linkedin.com/in/shreyasjha/), [Yaoyu Hu](http://www.huyaoyu.com/)
[Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Sebastian Scherer](https://theairlab.org/team/sebastian/), [Wenshan Wang](http://www.wangwenshan.com/)
</div>

## Updates
- [2025/06/10] Initial release of model checkpoint and inference code. 


## Overview

UFM(UniFlowMatch) is a simple, end-to-end trained transformer model that directly regresses pixel displacement image that applies concurrently to both optical flow and wide-baseline matching tasks. 

## Quick Start

First, recursively clone this repository and install the dependencies and the `UniCeption` library. It is a library contains modular, config-swappable components for assembling end-to-end vision networks.  

```
git clone --recursive https://github.com/UniFlowMatch/UFM.git
cd UFM

# In case you cloned without --recirsive:
# git submodule update --init

conda create -n ufm python=3.11 -y
conda activate ufm

# install UniCeption
cd UniCeption
pip install -e .
cd ..

# install uniflowmatch
pip install -r requirements.txt
pip install -e .
```

Then, verify your install by running

```bash
python uniflowmatch/models/ufm.py
```

Verify that `ufm_output.png` looks like `example/example_ufm_output.png`.

## Interactive Demo

## Citation
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{zhang2025ufm,
 title={UFM: A Simple Path towards Unified Dense Correspondence with Flow},
 author={Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
 booktitle={TBD},
 year={2025}
}
```