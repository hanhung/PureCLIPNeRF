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

from lib import utils, dvgo_imp
from lib.load_data import load_data

from DiffAugment_pytorch import DiffAugment

import jax


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0, cfg=None):
    '''Render images for the given viewpoints
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)

        factor = HW.reshape(-1)[0] / render_factor
        HW[:] = render_factor

        Ks = Ks.astype(float)
        Ks[:, :2, :3] /= factor

        HW = HW.astype(int)
        Ks = Ks.astype(int)

    rgbs = []
    depths = []
    alphas = []

    for i, c2w in enumerate(tqdm(render_poses)):
        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo_imp.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        alphas.append(render_result['alphainv_last'].cpu().numpy())
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs, depths


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    counter = 0
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo_imp.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        if counter == 0:
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
        else:
            xyz_min = torch.maximum(xyz_min, pts_nf.amin((0,1,2)))
            xyz_max = torch.minimum(xyz_max, pts_nf.amax((0,1,2)))
        counter += 1

    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, jax_key, coarse_ckpt_path=None, writer=None, base_step=0):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
        # init model
        model_kwargs = copy.deepcopy(cfg_model)
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        num_voxels = model_kwargs.pop('num_voxels')
        if len(cfg_train.pg_scale) and reload_ckpt_path is None:
            num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
        model = dvgo_imp.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
        model = model.to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model_class = dvgo_imp.DirectVoxGO
        model = utils.load_model(model_class, reload_ckpt_path).to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)

    """
    Split jax key
    """
    jax_key, _ = jax.random.split(jax_key)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'resolution': cfg.data.resolution,
        'num_bkgds': cfg.data.num_bkgds,
        'jax_key': jax_key,
    }

    # init batch rays sampler
    def gather_training_rays():
        """
        Change to new poses sampled each time
        """
        new_data_dict = load_data(cfg.data)
        if new_data_dict['irregular_shape']:
            new_data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in new_data_dict['images']]
        else:
            new_data_dict['images'] = torch.FloatTensor(new_data_dict['images'], device='cpu')
        new_data_dict['poses'] = torch.Tensor(new_data_dict['poses'])

        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
            new_data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
            ]
        ]

        if new_data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            raise NotImplementedError
        elif cfg_train.ray_sampler == 'flatten':
            raise NotImplementedError
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo_imp.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo_imp.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, _ = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        raise NotImplementedError

    """
    Init CLIP model and encode text descriptions
    """
    if not cfg_train.open_clip:
        clip_model, _ = clip.load(cfg_train.clip_model_name, device=device)
    else:
        clip_model, _, _ = open_clip.create_model_and_transforms(cfg_train.clip_model_name, pretrained=cfg_train.open_clip_pretrained)
        clip_model = clip_model.to(device)

    if cfg_train.clip_model_name_2 is not None:
        if not cfg_train.open_clip_2:
            clip_model_2, _ = clip.load(cfg_train.clip_model_name_2, device=device)
        else:
            clip_model_2, _, _ = open_clip.create_model_and_transforms(cfg_train.clip_model_name_2, pretrained=cfg_train.open_clip_pretrained_2)
            clip_model_2 = clip_model_2.to(device)

    """
    Normalizations for the image
    """
    norm_transform = transforms.Compose([
        transforms.RandomCrop((cfg.data.resolution - 16, cfg.data.resolution - 16)),
        transforms.Resize((cfg_train.clip_mode_res, cfg_train.clip_mode_res)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    norm_transform_2 = transforms.Compose([
        transforms.RandomCrop((cfg.data.resolution - 16, cfg.data.resolution - 16)),
        transforms.Resize((cfg_train.clip_mode_res_2, cfg_train.clip_mode_res_2)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    # Weird bug in mmcv with apostrophe, currently replacing prompts in validation with '#' and replacing here for quick fix
    corrected_text = cfg_train.query_text.replace("#", "'")

    """
    Encode text with the loaded CLIP models
    """
    with torch.no_grad():
        text = clip.tokenize(corrected_text).to(device)
        text_features = clip_model.encode_text(text)
        text_features = F.normalize(text_features, dim=1).unsqueeze(0)

        if cfg_train.clip_model_name_2 is not None:
            text_features_2 = clip_model_2.encode_text(text)
            text_features_2 = F.normalize(text_features_2, dim=1).unsqueeze(0)

    # GOGO
    torch.cuda.empty_cache()

    """
    Lists to record metrics and losses
    """
    kl_lst = []
    sim_lst = []
    sparsity_lst = []
    clip_loss_lst = []
    total_loss_lst = []
    time0 = time.time()
    global_step = -1

    """
    Get prior shape for density
    """
    spherical_prior = None
    grid_xyz = torch.stack(torch.meshgrid(
        torch.linspace(model.xyz_min[0], model.xyz_max[0], model.k0.shape[2]),
        torch.linspace(model.xyz_min[1], model.xyz_max[1], model.k0.shape[3]),
        torch.linspace(model.xyz_min[2], model.xyz_max[2], model.k0.shape[4]),
    ), -1)
    with torch.no_grad():
        radius = torch.sqrt((grid_xyz.unsqueeze(0).unsqueeze(1) ** 2).sum(-1))
        radius_mask = radius <= 1.0
        spherical_prior = torch.zeros((model.k0.shape[0], 1, model.k0.shape[2], model.k0.shape[3], model.k0.shape[4])).to(device)
        spherical_prior[radius_mask] = 1.0
        spherical_prior[~radius_mask] = 0.0

    """
    Initializations for other augmentations
    """
    if cfg_train.persp_aug:
        trans_strategy = transforms.RandomPerspective(distortion_scale=cfg_train.persp_distortion_scale, p=cfg_train.persp_p)
    else:
        trans_strategy = None

    for global_step in trange(1+start, 1+cfg_train.N_iters):
        # Get new poses every time
        if global_step % cfg.data.pose_refresh_rate == 0:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, _ = gather_training_rays()

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            with torch.no_grad():
                model_density = model.densitynet(model.k0.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            self_alpha = F.max_pool3d(model.activate_density(model_density), kernel_size=3, padding=1, stride=1)[0,0]
            model.mask_cache.mask &= (self_alpha > model.fast_color_thres)

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, dvgo_imp.DirectVoxGO):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

            """
            Get prior shape for density
            """
            spherical_prior = None
            grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(model.xyz_min[0], model.xyz_max[0], model.k0.shape[2]),
                torch.linspace(model.xyz_min[1], model.xyz_max[1], model.k0.shape[3]),
                torch.linspace(model.xyz_min[2], model.xyz_max[2], model.k0.shape[4]),
            ), -1)
            with torch.no_grad():
                radius = torch.sqrt((grid_xyz.unsqueeze(0).unsqueeze(1) ** 2).sum(-1))
                radius_mask = radius <= 1.0
                spherical_prior = torch.zeros((model.k0.shape[0], 1, model.k0.shape[2], model.k0.shape[3], model.k0.shape[4])).to(device)
                spherical_prior[radius_mask] = 1.0
                spherical_prior[~radius_mask] = 0.0

        """
        Since we render entire image sample all rays in one view
        """
        sel_b = np.random.choice(rgb_tr.shape[0], cfg.data.batch_size, replace=False)
        target = None
        rays_o = rays_o_tr[sel_b].reshape(-1, 3)
        rays_d = rays_d_tr[sel_b].reshape(-1, 3)
        viewdirs = viewdirs_tr[sel_b].reshape(-1, 3)

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)
        render_kwargs['jax_key'], _ = jax.random.split(render_kwargs['jax_key'])

        """
        Get and reshape the tranmittence, rgb and alphainv images
        """
        trans_mean = render_result['alphainv_last'].mean()
        rgb = render_result['rgb_marched'].unsqueeze(0).reshape(-1, cfg.data.resolution, cfg.data.resolution, 3).permute(0, 3, 1, 2)
        alphainv = render_result['alphainv_last'].unsqueeze(0).unsqueeze(-1).reshape(-1, 1, cfg.data.resolution, cfg.data.resolution)

        """
        Get augmentations from DiffAugment ('color,translation,resize,cutout')
        """
        if cfg_train.diff_aug:
            POLICY = 'color,translation,resize,cutout'
            rgb = rgb.unsqueeze(1).repeat(1, cfg.data.num_diffs, 1, 1, 1)
            rgb = rgb.reshape(-1, 3, cfg.data.resolution, cfg.data.resolution)
            rgb = DiffAugment(rgb, policy=POLICY)

        """
        Combine background augmentation with image, rescale image for CLIP model and get mean transmittence
        """
        if cfg_train.bkgd_aug:
            bkgds = np.stack(render_result['bg'], axis=0)
            if cfg_train.blur_bkgd:
                min_blur, max_blur = cfg_train.bg_blur_std_range
                blur_std = np.random.rand(1) * (max_blur - min_blur) + min_blur
                blur_std = blur_std[0]

                bkgds = bkgds.reshape(-1, cfg.data.resolution, cfg.data.resolution, 3)
                for i in range(bkgds.shape[0]):
                    bkgds[i] = cv2.GaussianBlur(bkgds[i], [15, 15], blur_std, blur_std, cv2.BORDER_DEFAULT)
                bkgds = bkgds.reshape(bkgds.shape[0], cfg.data.resolution * cfg.data.resolution, 3)

            bkgds = torch.from_numpy(bkgds).float().to(device)
            bkgds = bkgds.reshape(-1, cfg.data.resolution, cfg.data.resolution, 3).permute(0, 3, 1, 2)

            bkgds = bkgds.unsqueeze(1) * alphainv.unsqueeze(0)
            rgb = rgb.unsqueeze(0) + bkgds
            rgb = rgb.reshape(-1, 3, cfg.data.resolution, cfg.data.resolution)
        
        if trans_strategy is not None:
            rgb = trans_strategy(rgb)

        rgb_norm = norm_transform(rgb)

        """
        Get image features from CLIP model and calculate clip similarity loss
        """
        image_features = clip_model.encode_image(rgb_norm)
        image_features = F.normalize(image_features, dim=1).unsqueeze(0)

        """
        If using a second clip model
        """
        if cfg_train.clip_model_name_2 is not None:
            rgb_norm_2 = norm_transform_2(rgb)
            image_features_2 = clip_model_2.encode_image(rgb_norm_2)
            image_features_2 = F.normalize(image_features_2, dim=1).unsqueeze(0)

        loss = 0

        similarity = (image_features * text_features).sum(-1).mean()
        clip_loss = 1 - similarity
        if global_step >= cfg_train.clip_model_start and global_step <= cfg_train.clip_model_end:
            loss += clip_loss

        if cfg_train.clip_model_name_2 is not None and global_step >= cfg_train.clip_model_start_2 and global_step <= cfg_train.clip_model_end_2:
            similarity_2 = (image_features_2 * text_features_2).sum(-1).mean()
            clip_loss_2 = 1 - similarity_2
            loss += clip_loss_2 * cfg_train.clip_model_weight_2

        """
        Get the annealed target opacity
        """
        def anneal_exponentially(cur_step, total_step, val0, val1):
            t = min(cur_step / total_step, 1.)
            val0 = torch.tensor(val0).float()
            val1 = torch.tensor(val1).float()
            return torch.exp(torch.log(val0) * (1 - t) + torch.log(val1) * t)
        opacity = anneal_exponentially(base_step + global_step, cfg_train.anneal_iterations, cfg_train.opacity_target0, cfg_train.opacity_target1).float().to(device)
        tau_sparsity = 1 - opacity

        """
        Calculate sparsity loss over the average transmittence
        """
        loss_sparsity = -torch.clamp(trans_mean, max=tau_sparsity)
        loss += cfg_train.lambda_sparsity * loss_sparsity

        """
        KL Divergence for prior loss on density
        """
        pred_prob = model.densitynet(model.k0.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        pred_prob = model.activate_density(pred_prob)
        loss_kl = (spherical_prior * (spherical_prior / (pred_prob + 1e-4) + 1e-4).log()).mean()
        if global_step >= cfg_train.prior_after and global_step <= cfg_train.prior_before:
            loss += cfg_train.prior_loss_weight * loss_kl

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss


        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                loss += cfg_train.weight_tv_density * model.density_total_variation()
            if cfg_train.weight_tv_k0>0:
                loss += cfg_train.weight_tv_k0 * model.k0_total_variation()
        
        loss.backward()
        optimizer.step()

        """
        Gather loss and metric information into lists
        """
        kl_lst.append(loss_kl.item())
        sim_lst.append(similarity.item())
        total_loss_lst.append(loss.item())
        clip_loss_lst.append(clip_loss.item())
        sparsity_lst.append(cfg_train.lambda_sparsity * loss_sparsity.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        """
        Log gpu memory usage
        """
        if global_step%100==0:
            def get_gpu_memory():
                output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
                ACCEPTABLE_AVAILABLE_MEMORY = 1024
                COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
                try:
                    memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
                except sp.CalledProcessError as e:
                    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
                memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
                return memory_use_values
            
            if writer is not None and 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
                memory_usage = get_gpu_memory()[int(gpu_id)]
                writer.add_scalar('GPU/Memory Usage (MiB)', memory_usage, global_step + base_step)
                writer.add_scalar('Scalar/voxel res x', model.world_size[0].item(), global_step + base_step)
                writer.add_scalar('Scalar/voxel res y', model.world_size[1].item(), global_step + base_step)
                writer.add_scalar('Scalar/voxel res z', model.world_size[2].item(), global_step + base_step)

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / Similarity: {np.mean(sim_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            if writer is not None:
                writer.add_scalar('Loss/total_loss', np.mean(total_loss_lst), global_step + base_step)
                writer.add_scalar('Loss/clip_loss', np.mean(clip_loss_lst), global_step + base_step)
                writer.add_scalar('Loss/sparsity_loss', np.mean(sparsity_lst), global_step + base_step)
                writer.add_scalar('Metric/similarity', np.mean(sim_lst), global_step + base_step)
                writer.add_scalar('Loss/kl_loss', np.mean(kl_lst), global_step + base_step)
                writer.add_image('rgb', render_result['rgb_marched'].detach().reshape(-1, cfg.data.resolution, cfg.data.resolution, 3).permute(0, 3, 1, 2)[0].cpu().numpy(), global_step + base_step)
                writer.add_image('aug_rgb', rgb[0].detach().squeeze(0).cpu().numpy(), global_step + base_step)

                rgb8 = utils.to8b(render_result['rgb_marched'].detach().reshape(-1, cfg.data.resolution, cfg.data.resolution, 3)[0].cpu().numpy())
                filename = os.path.join(cfg.basedir, cfg.expname, 'exp', '{}.png'.format(global_step + base_step))
                imageio.imwrite(filename, rgb8)

            kl_lst = []
            sim_lst = []
            sparsity_lst = []
            clip_loss_lst = []
            total_loss_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict, jax_key, writer=None):    
    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    """
    Split jax key
    """
    jax_key, _ = jax.random.split(jax_key)

    # fine detail reconstruction
    eps_fine = time.time()
    xyz_min_fine, xyz_max_fine = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine', jax_key=jax_key,
            coarse_ckpt_path=None, writer=writer, base_step=1)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')
