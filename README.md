# Guiding Text-to-Image Diffusion Model Towards Grounded Generation

This repository contains the official PyTorch implementation of grounded diffusion: https://arxiv.org/abs/2301.05221.

## Requirements
A suitable [conda](https://conda.io/) environment named `grounded-diffusion` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate grounded-diffusion
```

## Model Zoo

https://drive.google.com/drive/folders/1HlagN6jVhmC_UbrOAy133LkN4Qgf2Scv?usp=sharing

## Train

Comming soon....

## Inference
```bash
python test.py --sd_ckpt 'xxx/stable_diffusion.ckpt' \
--grounding_ckpt 'xxx/grounding_module.pth' \
--prompt "a photo of a lion on a mountain top at sunset" \
--category "lion"

```
## Citation
If you use this code for your research or project, please cite:

	@article{li2023grounded,
	  title={Guiding Text-to-Image Diffusion Model Towards Grounded Generation},
	  author={Li, Ziyi and Zhou, Qinye and Zhang, Xiaoyun and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
	  journal = {arXiv:2301.05221},
	  year={2023}
	}
	
## Acknowledgements
Many thanks to the code bases from [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [CLIP](https://github.com/openai/CLIP).
