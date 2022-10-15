# Pure CLIP NeRF

<img src="./figures/teaser/Fig-Teaser.png">

Initial code release for the paper [**Understanding Pure CLIP Guidance for Voxel Grid NeRF Models**](https://arxiv.org/abs/2209.15172).

## Installation

*We have tested our scripts using PyTorch 1.11 with Cuda 11.3 on Ubuntu 20.04.*

1. Create Conda environment.
```
$ conda create -n PureCLIPNeRF python=3.8
$ conda activate PureCLIPNeRF
```
2. Install PyTorch.
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
3. Install packages required by DVGO.
```
$ pip install -r requirements.txt
```
4. Install torch scatter.
```
$ conda install pytorch-scatter -c pyg
```
5. Install jax related libraries (cpu version is fine, jax is only used to generate background augmentations).
```
$ pip install --upgrade pip
$ pip install --upgrade "jax[cpu]"
$ pip install flax==0.5.3
$ pip install dm_pix
```
6. Install CLIP and OpenCLIP.
```
$ pip install git+https://github.com/openai/CLIP.git
$ pip install open_clip_torch
```

## Training
```
$ python run.py --config configs/low/exp_vit16.py --prompt "matte painting of a bonsai tree; trending on artstation."
$ python run.py --config configs/low/imp_vit16.py --prompt "matte painting of a bonsai tree; trending on artstation."
```
### Config File Naming
* exp_\*.py, imp_\*.py: Explicit and Implicit voxel grid models respectively.
* \*_vit16.py: Trained with CLIP ViT-B/16 model.

### Config File Folders
* configs/low/\*: Settings that will run on GPUs with at least 11GB of VRAM. (tested on RTX 2080 Ti)
* configs/mid/\*: Settings that will run on GPUs with at least 24GB of VRAM. (tested on RTX 3090)
* configs/paper/\*: Settings used in the paper. (tested on RTX A6000)

## Config File
### Guidance Models
1. To change OpenAI CLIP models, change the clip model name and set the image resolution of the CLIP model. Following the naming convention from: https://github.com/openai/CLIP.
```
clip_model_name = 'ViT-B/16',
clip_mode_res = 224,
```
2. To use OpenCLIP models, change the clip model name and set the image resolution of the CLIP model. Following the naming convention from: https://github.com/mlfoundations/open_clip. 
```
clip_model_name = 'ViT-B-16-plus-240',
clip_mode_res = 240,
open_clip = True,
open_clip_pretrained = 'laion400m_e32',
```
### Ensemble Models
1. To ensemble CLIP models, enter the clip model name and set the image resolution of the CLIP model in the second slot.
```
clip_model_name = 'ViT-B/32',
clip_mode_res = 224,
clip_model_start = 0,
clip_model_end = 40000,
open_clip = False,
open_clip_pretrained = None,

clip_model_name_2 = 'ViT-L/14',
clip_model_weight_2 = 0.5,
clip_mode_res_2 = 224,
clip_model_start_2 = 5000,
clip_model_end_2 = 40000,
open_clip_2 = False,
open_clip_pretrained_2 = None,
```

## Acknowledgements
[**DVGO**](https://github.com/sunset1995/DirectVoxGO): Our backbones are heavily based on DVGO and their implementation.

[**Dream Fields**](https://github.com/google-research/google-research/tree/master/dreamfields): We use their code for background augmentations in *lib/jax_bkgd* and reimplement losses.

[**DiffAugment**](https://github.com/mit-han-lab/data-efficient-gans): We use DiffAugment from their code in *DiffAugment_pytorch.py*.

[**CLIP**](https://github.com/openai/CLIP), [**OpenCLIP**](https://github.com/mlfoundations/open_clip): We use both CLIP and OpenCLIP models for guidance in our models.

Thanks to the authors for the awesome works above and releasing their code! Please check out their papers for more details.

## TO-DO
- [X] Add remaining paper configs.
- [X] Add more OpenCLIP configs.
- [ ] Add section about tuning voxel grid resolution and scheduling.
- [ ] Add figures showing difference between low, mid and high.
- [ ] Mask unneeded forward passes for implicit model.
- [ ] Deferred rendering to save memory.
