try:
    from cStringIO import StringIO as BytesIO
except: # Python 3
    from io import BytesIO
import numpy as np
import PIL.Image
import shutil
from math import sqrt
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.exposure import adjust_gamma


def save_image_array(img, filename, resize_multiplier=(1, 1), gamma_multiplier=1, rescale_values=True, flipud=True, flat_wide=False, flat_vert=False, fmt="png"):
    """
    Expects a 4D image array of (n_images, height, width, channels)

    rescale will rescale 1 channel images to the maximum value available

    Modified from implementation by Kyle McDonald

    https://github.com/kylemcdonald/python-utils/blob/master/show_array.py
    """

    if len(img.shape) != 4:
       raise ValueError("Expects a 4D image array of (n_images, height, width, channels)")

    if flipud:
        img = img[:, ::-1]

    n_ex, o_height, o_width, o_channels = img.shape

    if img.shape[0] != 1:
        n = len(img)
        side = int(sqrt(n))
        side0 = side
        side1 = side
        shp = img.shape
        if flat_wide or flat_vert or (side * side) == n:
            pass
        else:
            raise ValueError("Need input length that can be reshaped to a square (4, 16, 25, 36, etc)")
        n,h,w,c = img.shape
        if flat_wide:
            assert flat_wide != flat_vert
            side0 = 1
            side1 = n_ex
        elif flat_vert:
            assert flat_wide != flat_vert
            side0 = n_ex
            side1 = 1
        img = img.reshape(side0, side1, h, w, c).swapaxes(1, 2).reshape(side0*h, side1*w, c)
    else:
        img = img[0]

    if rescale_values:
        """
        img_max = np.max(img)
        img_min = np.min(img)
        # scale to 0, 1
        img = (img - img_min) / float(img_max - img_min)
        # scale 0, 1 to 0, 255
        """
        img *= 255.

    if img.shape[-1] == 1:
       img = img[:, :, 0]

    img = np.uint8(np.clip(img, 0, 255))
    if resize_multiplier != (1, 1):
        rs = resize(img, (img.shape[0] * resize_multiplier[0], img.shape[1] * resize_multiplier[1]))

    if gamma_multiplier != 1:
        rs = adjust_gamma(rs, gamma_multiplier)

    if resize_multiplier != (1, 1) or gamma_multiplier != 1:
        rs *= 255.
        img = np.uint8(np.clip(rs, 0, 255))
    image_data = BytesIO()
    PIL.Image.fromarray(img).save(image_data, fmt)
    with open(filename, 'wb') as f:
        image_data.seek(0)
        shutil.copyfileobj(image_data, f)
