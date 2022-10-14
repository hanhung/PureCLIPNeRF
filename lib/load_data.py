import math
import numpy as np

from .load_camera import sample_cameras

def load_data(args):

    K, depths = None, None

    if args.sample_cameras:
        images, poses, render_poses, hwf, i_split = sample_cameras(args.datadir, args.half_res, args.testskip, args.resolution, args.num_sampled_poses)
        near = 4. - math.sqrt(3) * 1
        far = 4. + math.sqrt(3) * 1

        # print('Sampling cameras from prior')
        i_train, i_val, i_test = i_split

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        HW = np.array([im.shape[:2] for im in images])
        irregular_shape = False

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
    else:
        raise NotImplementedError

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
    )
    return data_dict

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far
