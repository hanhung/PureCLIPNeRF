# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# From: https://github.com/google-research/google-research/tree/master/dreamfields

"""Data augmentation helpers with random crops and backgrounds."""

from lib.jax_bkgd import optvis

import dm_pix
import jax
from jax import random
import jax.numpy as np
import numpy as onp

def checkerboard(key, blur_key, nsq, size, bg_blur_std_range=None, dtype=np.float32):
    """Create a checkerboard background image with random colors.

    NOTE: only supports a single value for nsq (number squares).

    Args:
    key: JAX PRNGkey.
    nsq (int): number of squares per side of the checkerboard.
    size (int): size of one side of the checkerboard in pixels.
    dtype: desired return data type.

    Returns:
    canvas (np.array): checkerboard background image.
    """
    assert size % nsq == 0
    sq = size // nsq
    color1, color2 = random.uniform(key, (2, 3), dtype=dtype)
    canvas = np.full((nsq, sq, nsq, sq, 3), color1, dtype=dtype)
    canvas = jax.ops.index_update(canvas, jax.ops.index[::2, :, 1::2, :, :], color2)
    canvas = jax.ops.index_update(canvas, jax.ops.index[1::2, :, ::2, :, :], color2)
    bg = canvas.reshape(sq * nsq, sq * nsq, 3)
    if bg_blur_std_range is not None:
        min_blur, max_blur = bg_blur_std_range
        blur_std = random.uniform(blur_key) * (max_blur - min_blur) + min_blur
        bg = dm_pix.gaussian_blur(bg, blur_std, kernel_size=15)
    return bg

def noise(key, blur_key, size, bg_blur_std_range=None):
    bg = random.uniform(key, (size, size, 3))
    if bg_blur_std_range is not None:
        min_blur, max_blur = bg_blur_std_range
        blur_std = random.uniform(blur_key) * (max_blur - min_blur) + min_blur
        bg = dm_pix.gaussian_blur(bg, blur_std, kernel_size=15)
    return bg

def fft(key, blur_key, size, bg_blur_std_range=None):
    bg = optvis.image_sample(key, [1, size, size, 3], sd=0.2, decay_power=1.5)[0]
    if bg_blur_std_range is not None:
        min_blur, max_blur = bg_blur_std_range
        blur_std = random.uniform(blur_key) * (max_blur - min_blur) + min_blur
        bg = dm_pix.gaussian_blur(bg, blur_std, kernel_size=15)
    return bg
