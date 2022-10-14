import subprocess as sp
from shutil import copyfile
from tqdm import tqdm, trange
import os, sys, copy, glob, json, time, random, argparse

import cv2
import mmcv
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

import clip
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import train_exp, train_imp
from lib import utils, dvgo_exp, dvgo_imp
from lib.load_data import load_data

from DiffAugment_pytorch import DiffAugment

import jax

def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_voxel_grid_only", type=str, default='',
                        help='export voxel grids for debugging and 3d visualization')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    # Manual text prompt input
    parser.add_argument("--prompt", type=str, default=None,
                        help='input text prompt for text to 3d')
    return parser

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

"""
Setup jax keys for background augmentation
"""
jax.config.update('jax_platform_name', 'cpu')
jax_key = jax.random.PRNGKey(0)

# load setup
parser = config_parser()
args = parser.parse_args()
cfg = mmcv.Config.fromfile(args.config)

if args.prompt is not None:
    cfg.fine_train.query_text = args.prompt

# init enviroment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
seed_everything()

# load images / poses / camera settings / data split
data_dict = load_everything(args=args, cfg=cfg)

if cfg.fine_train.mode == 'implicit':
    dvgo = dvgo_imp
    train = train_imp.train
    render_viewpoints = train_imp.render_viewpoints
else:
    dvgo = dvgo_exp
    train = train_exp.train
    render_viewpoints = train_exp.render_viewpoints

# export scene bbox and camera poses in 3d for debugging and visualization
if args.export_bbox_and_cams_only:
    print('Export bbox and cameras...')
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
    near, far = data_dict['near'], data_dict['far']
    cam_lst = []
    for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
        cam_o = rays_o[0,0].cpu().numpy()
        cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
        cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
    np.savez_compressed(args.export_bbox_and_cams_only,
        xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
        cam_lst=np.array(cam_lst))
    print('done')
    sys.exit()

if args.export_voxel_grid_only:
    print('Export voxel grid visualization...')
    with torch.no_grad():
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)

        if cfg.fine_train.mode == 'implicit':
            model_density = model.densitynet(model.k0.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            model_rgb = model.rgbnet(model.k0.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            alpha = model.activate_density(model_density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model_rgb).squeeze().permute(1,2,3,0).cpu().numpy()
        else:
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
    np.savez_compressed(args.export_voxel_grid_only, alpha=alpha, rgb=rgb)
    print('done')
    sys.exit()

# train
if not args.render_only:
    writer = SummaryWriter(os.path.join(cfg.basedir, cfg.expname, 'exp'))
    train(args, cfg, data_dict, jax_key, writer)

# load model for rendering
ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
ckpt_name = ckpt_path.split('/')[-1][:-4]
model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
model_class = dvgo.DirectVoxGO
model = utils.load_model(model_class, ckpt_path).to(device)
stepsize = cfg.fine_model_and_render.stepsize
render_viewpoints_kwargs = {
    'model': model,
    'ndc': cfg.data.ndc,
    'render_kwargs': {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'resolution': cfg.data.resolution,
        'num_bkgds': cfg.data.num_bkgds,
        'jax_key': jax_key,
        'render_depth': True,
    },
}

# render testset
if args.render_test:
    testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
    os.makedirs(testsavedir, exist_ok=True)
    rgbs, depths = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir, render_factor=224, cfg=cfg,
            **render_viewpoints_kwargs)

testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
os.makedirs(testsavedir, exist_ok=True)
rgbs, depths = render_viewpoints(
        render_poses=data_dict['render_poses'],
        HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
        Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
        savedir=testsavedir, render_factor=cfg.data.render_resolution, cfg=cfg,
        **render_viewpoints_kwargs)
imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

print('Done')
