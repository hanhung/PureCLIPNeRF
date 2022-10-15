from copy import deepcopy

expname = 'mid_imp_open_vit16'    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=True,              # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=False,               # [TODO]

    # Below are forward-facing llff specific settings. Not support yet.
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    factor=4,                     # [TODO]
    width=None,                   # enforce image width
    height=None,                  # enforce image height
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth

    resolution=224,               # Training time render resolution
    render_resolution=512,        # Render resolution at inference time

    num_diffs=4,                  # Number of Diff Augmentations to apply
    num_bkgds=8,                  # Number of Background Augmentations to apply

    batch_size=1,                 # Number of images rendered ber optimization iteration
    sample_cameras=True,
    num_sampled_poses=1000,
    pose_refresh_rate=5000,
)

'''
Fixed base train settings
'''
base_train = dict(
    mode='implicit',              # Whether to use implicit or explicit voxel grid

    N_rand=8192,                  # batch size (number of random rays per optimization step)

    lrate_density=0.0,            # lr of density voxel grid
    lrate_k0=0.0,                 # lr of color/feature voxel grid
    lrate_rgbnet=5e-3,            # lr of the mlp to preduct view-dependent color
    lrate_densitynet=5e-3,        # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps

    pervoxel_lr=False,            # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_entropy_last=0.01,     # weight of background entropy loss
    skip_zero_grad_fields=\
        ['density', 'k0'],        # the variable name to skip optimizing parameters w/ zero grad in each iteration

    bkgd_aug=True,                # whether to add background augmentation
    blur_bkgd=True,               # whether to blur the background augmentations
    lambda_sparsity=0.5,          # weight for the sparisty loss
    opacity_target0 = 0.5,        # sparsity target at beginning
    opacity_target1 = 0.1,        # final sparsity target
    anneal_iterations = 500,      # how many iterations to anneal to the sparsity
    bg_blur_std_range=[0., 10.],  # blur range for the blur background augmentations

    diff_aug = True,              # Whether to add diff augmentation
)

'''
Tune fine train settings for different train settings
'''
fine_train = deepcopy(base_train)
fine_train.update(dict(
    N_iters=10000,                # number of optimization steps
    pg_scale=[4000, 6000, 8000],  # checkpoints for progressive scaling

    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=4000,                # count total variation loss from tv_from step
    tv_before=9000,               # count total variation before the given number of iterations
    tv_dense_before=0,            # count total variation densely before the given number of iterations
    weight_tv_density=0.2,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.2,             # weight of total variation loss of color/feature voxel grid

    prior_after = 0,              # Iteration to start adding kl loss
    prior_before = 7000,          # Iteration to stop adding kl loss
    prior_loss_weight = 0.2,      # Lambda weight for kl loss

    persp_aug = False,            # Whether to turn on perspective augmentation
    persp_p = 1.0,                # Probability for random perspective
    persp_distortion_scale = 0.6, # Distortion rate for random perspective

    clip_model_name = 'ViT-B-16', # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    clip_mode_res = 224,
    clip_model_start = 0,
    clip_model_end = 40000,
    open_clip = True,
    open_clip_pretrained = 'laion400m_e32',

    clip_model_name_2 = None,
    clip_model_weight_2 = 0.5,
    clip_mode_res_2 = None,
    clip_model_start_2 = 0,
    clip_model_end_2 = 40000,
    open_clip_2 = False,
    open_clip_pretrained_2 = None,

    query_text="The cat is sleeping comfortably on the chair.",
))

'''
Fixed model and rendering settings
'''
base_model_and_render = dict(
    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-4,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=False,   # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=0.5,                 # sampling stepsize in volume rendering
)

fine_model_and_render = deepcopy(base_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=125**3,
    num_voxels_base=125**3,
    activation='relu',
))

del deepcopy
