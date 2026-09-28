"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from functools import partial
import random as rnd
import numpy as np
from PIL import ImageFilter, Image
from timm.data import auto_augment

_OP_CACHE = {}

def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))

def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)

# The three noise/blur ops below replace imgaug (MotionBlur, AdditiveGaussianNoise,
# AdditivePoissonNoise), which does not import under NumPy 2.

def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    angle = rnd.uniform(0, 180)
    key = f'motion_blur_{k}_{int(angle)}'
    def make_kernel():
        kernel = Image.new('L', (k, k), 0)
        from PIL import ImageDraw
        c = (k - 1) / 2
        dx, dy = c * np.cos(np.radians(angle)), c * np.sin(np.radians(angle))
        ImageDraw.Draw(kernel).line([(c - dx, c - dy), (c + dx, c + dy)], fill=255)
        weights = np.asarray(kernel, dtype=np.float64).ravel()
        return ImageFilter.Kernel((k, k), weights.tolist(), scale=weights.sum() or 1) if k <= 5 else None
    op = _get_op(key, make_kernel)
    if op is None:  # PIL kernels are limited to 5x5: approximate larger blurs with a box blur
        return img.filter(ImageFilter.BoxBlur(k // 4 or 1))
    return img.filter(op)

def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, scale, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    arr = np.asarray(img).astype(np.float32)
    # imgaug's AdditivePoissonNoise adds Poisson(lam) samples with a random sign per pixel
    arr += np.random.poisson(lam, arr.shape) * np.random.choice([-1.0, 1.0], arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def salt_and_pepper_noise(image, prob=0.05):
    if prob <= 0:
        return image
    arr = np.asarray(image)
    original_dtype = arr.dtype
    intensity_levels = 2 ** (arr[0, 0].nbytes * 8)
    min_intensity = 0
    max_intensity = intensity_levels - 1
    random_image_arr = np.random.choice([min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape)
    salt_and_peppered_arr = arr.astype(np.float64) * random_image_arr
    salt_and_peppered_arr = np.nan_to_num(salt_and_peppered_arr, nan=max_intensity).astype(original_dtype)
    return Image.fromarray(salt_and_peppered_arr)

def random_border_crop(image):
    img_width,img_height = image.size
    crop_left = int(img_width * rnd.uniform(0.0, 0.025))
    crop_top = int(img_height * rnd.uniform(0.0, 0.075))            
    crop_right = int(img_width * rnd.uniform(0.975, 1.0))
    crop_bottom = int(img_height * rnd.uniform(0.925, 1.0))
    final_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return final_image

def random_rotation(image, max_deg=5, max_shift=0.25):
    """Small rotation that keeps the whole line in frame. The angle is capped so the ends of the
    line move by at most max_shift x its height: for a full printed line (20x wider than tall)
    that is under 1 degree, while a fixed 5 degrees would push the ends out of the image."""
    w, h = image.size
    limit = min(max_deg, np.degrees(np.arctan(max_shift * h / max(w, 1))))
    arr = np.asarray(image)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    fill = np.median(border, axis=0)
    fill = tuple(int(v) for v in fill) if arr.ndim == 3 else int(fill)   # background, even if inverted
    return image.rotate(rnd.uniform(-limit, limit), resample=Image.BICUBIC, expand=True, fillcolor=fill)

def random_resize(image):
    size = image.size
    new_size = [rnd.randint(int(0.5*size[0]), int(1.5*size[0])), rnd.randint(int(0.5*size[1]), int(1.5*size[1]))]
    reduce_factor = rnd.randint(1,4)
    new_size = tuple([int(x/reduce_factor) for x in new_size])
    final_image = image.resize(new_size)
    return final_image

def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return level,

_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    # 'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
]
#_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    'GaussianNoise',
    'PoissonNoise'
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40)
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise
})

def rand_augment_transform(magnitude=5, num_layers=3):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'img_mean':128,
        # 'rotate_deg': 5,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.0,
    }
    # keywords, not positions: timm >= 0.9 added a `prob` parameter before `hparams`
    ra_ops = [auto_augment.AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
              for name in _RAND_TRANSFORMS]
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1. / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)