import os
import cv2
import json
import imageio
import numpy as np

import torch
import torch.nn.functional as F

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def scale_intrinsics(new_width):
    """Scale camera intrinsics (heigh, width focal) to a desired image width."""
    # Height, width and focal length of the image plane in pixels, from NeRF's
    # Blender dataset of realistic synthetic objects. The focal length determines
    # how zoomed in renderings appear.
    hwf = np.array([800., 800., 1111.1111])
    return hwf * new_width / hwf[1]

def pose_spherical_opengl(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def sample_cameras(basedir, half_res=False, testskip=1, resolution=None, num_poses=200):
    splits = ['train', 'val', 'test']

    num_train = num_poses
    num_val = 1              # dummy variable
    num_test = 1

    th_range = [-180, 180]
    phi_range = [-30, -30]
    rad_range = [4., 4.]
    focal_mult_range = [1.2, 1.2]

    th = np.random.rand(num_train + num_val + num_test) * (th_range[1] - th_range[0]) + th_range[0]
    phi = np.random.rand(num_train + num_val + num_test) * (phi_range[1] - phi_range[0]) + phi_range[0]
    rad = np.random.rand(num_train + num_val + num_test) * (rad_range[1] - rad_range[0]) + rad_range[0]
    focal_mult = np.random.rand(num_train + num_val + num_test) * (focal_mult_range[1] - focal_mult_range[0]) + focal_mult_range[0]

    ####################################################################################################################################################
    # Holdout pose for test of elevation 45 degrees
    th[-1] = 30
    phi[-1] = -45
    ####################################################################################################################################################

    poses = torch.stack([pose_spherical(th[i], phi[i], rad[i]) for i in range(th.shape[0])], 0)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    counts = [0, num_train, num_train + num_val, num_train + num_val + num_test]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    hwf = scale_intrinsics(resolution)

    return np.zeros((num_train + num_val + num_test, resolution, resolution, 4)), poses, render_poses, hwf, i_split
