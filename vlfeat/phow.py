from __future__ import division, print_function
import warnings
import numpy as np

from .imsmooth import vl_imsmooth
from .dsift import vl_dsift
from .utils import as_float_image, rgb2hsv, rgb2gray

COLOR_CHOICES = ['gray', 'rgb', 'hsv', 'opponent']
DEFAULT_COLOR = 'gray'
DEFAULT_SIZES = (4, 6, 8, 10)
DEFAULT_STEP = 2
DEFAULT_MAGNIF = 6
DEFAULT_WINDOW_SIZE = 1.5
DEFAULT_CONTRAST_THRESH = 0.005


def vl_phow(image, fast=True, sizes=DEFAULT_SIZES, step=DEFAULT_STEP,
            color=DEFAULT_COLOR, contrast_thresh=DEFAULT_CONTRAST_THRESH,
            window_size=DEFAULT_WINDOW_SIZE, magnif=DEFAULT_MAGNIF,
            float_descriptors=False, verbose=False):
    if color not in COLOR_CHOICES:
        raise ValueError("unknown color {!r}; expected one of {}".format(
            color, ', '.join(repr(x) for x in COLOR_CHOICES)))

    dsift_opts = {
        'norm': True,
        'window_size': window_size,
        'fast': fast,
        'float_descriptors': float_descriptors,
        'step': step,
        'verbose': verbose,
    }

    # standardize the image
    if not 2 <= image.ndim <= 3:
        raise TypeError("image should be 2d or 3d")
    image = as_float_image(image, order='F')

    assert 2 <= image.ndim <= 3
    if image.ndim == 3 and image.shape[2] == 4:
        warnings.warn("ignoring alpha channel")
        image = image[:, :, :3]

    if color == 'gray':
        channels = 1
        if image.ndim == 3 and image.shape[2] > 1:
            image = rgb2gray(image)
        image = image.reshape(image.shape[:2] + (1,))
    else:
        channels = 3
        if image.ndim == 2 or image.shape[2] == 1:
            warnings.warn("asked for color features from a grayscale image")
            image = np.dstack([image] * 3)

        assert image.ndim == 3
        assert image.shape[2] == 3

        if color == 'hsv':
            image = rgb2hsv(image)
        elif color == 'rgb':
            pass
        else:
            # Note that the mean differs from the standard def. of opponent
            # space and is the regular intensity (for compatibility with
            # the contrast thresholding).
            #
            # Note also that the mean is added pack to the other two
            # components with a small multipliers for monochromatic
            # regions.
            r, g, b = np.rollaxis(image, axis=-1)
            mu = 0.3 * r + 0.59 * g + 0.11 * b
            alpha = 0.01
            image = np.dstack([
                mu,
                (r - g) / np.sqrt(2) + alpha * mu,
                (r + g - 2 * b) / np.sqrt(2) + alpha * mu,
            ])
            del r, g, b, mu, alpha

    if verbose:
        pr = lambda *a, **k: print('vl_phow:', *a, **k)
        pr('color space: {}'.format(color))
        pr('image size: {} x {}'.format(*image.shape[:2]))
        pr('sizes: [{}]'.format(', '.join(map(str, sizes))))

    frames = []
    descrs = []
    max_size = max(sizes)
    for size_i, size in enumerate(sizes):
        # Recall from vl_dsift() that the first descriptor for scale size has
        # center located at xc = xmin + 3/2 size (the y coordinate is
        # similar). It is convenient to align the descriptors at different
        # scales so that they have the same geometric centers. For the
        # maximum size we pick xmin = 1 and we get centers starting from
        # xc = 1 + 3/2 max(sizes). For any other scale we pick xmin so
        # that xmin + 3/2 size = 1 + 3/2 max(sizes).
        #
        # In practice, the offset must be integer ('bounds'), so the
        # alignment works properly only if all sizes are even or odd.

        offset = int(np.floor(1 + 1.5 * (max_size - size)))
        bounds = [offset - 1, offset - 1, np.inf, np.inf]

        # smooth the image to the appropriate scale based on the size of the
        # SIFT bins
        sigma = size / magnif
        smoothed = vl_imsmooth(image, sigma)

        # extract dense SIFT features from each channel
        f, d = zip(*[
            vl_dsift(smoothed[:, :, k], size=size, bounds=bounds, **dsift_opts)
            for k in range(channels)
        ])

        if d[0].size == 0:
            warnings.warn("didn't get any features at size {}".format(size))
            continue

        # zero out low-contrast descriptors
        # note that for HSV descriptors, the V component is thresholded
        if color in ('gray', 'opponent'):
            contrast = f[0][:, 2]
        elif color == 'hsv':
            contrast = f[2][:, 2]
        else:  # rgb
            contrast = np.mean([f_chan[:, 2] for f_chan in f], axis=0)

        # d = [d_chan[:, contrast < contrast_thresh] for d_chan in d]
        # f = [f_chan[:, contrast < contrast_thresh] for f_chan in f]
        for k in range(channels):
            d[k][contrast < contrast_thresh, :] = 0

        # save frames' positions, norms, and scale
        frames.append(np.hstack([f[0], size * np.ones((f[0].shape[0], 1))]))
        descrs.append(np.hstack(d))

    if not len(frames):
        warnings.warn("didn't get any features for image")
        return np.zeros((0, 4)), np.zeros((0, 128 * channels))
    return np.vstack(frames), np.vstack(descrs)
