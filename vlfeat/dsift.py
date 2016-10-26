from __future__ import division, print_function

import numpy as np
import numpy.ctypeslib as npc

from ctypes import cast, POINTER, Structure, c_int, c_float, c_double
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

from .vl_ctypes import load_library, c_to_np_types
from .utils import as_float_image, check_integer, check_float

LIB = load_library()

np_float = c_to_np_types[c_float]
np_double = c_to_np_types[c_double]

class VLDsiftKeypoint(Structure):
    _fields_ = [
        ('x', c_double),
        ('y', c_double),
        ('s', c_double),
        ('norm', c_double),
    ]

class VLDsiftDescriptorGeometry(Structure):
    _fields_ = [
        ('numBinT', c_int),
        ('numBinX', c_int),
        ('numBinY', c_int),

        ('binSizeX', c_int),
        ('binSizeY', c_int),
    ]

class VLDsiftFilter(Structure):
    _fields_ = [
        ('imWidth', c_int),
        ('imHeight', c_int),

        ('stepX', c_int),
        ('stepY', c_int),

        ('boundMinX', c_int),
        ('boundMinY', c_int),
        ('boundMaxX', c_int),
        ('boundMaxY', c_int),

        ('geom', VLDsiftDescriptorGeometry),

        ('useFlatWindow', c_int),
        ('windowSize', c_double),

        ('numFrames', c_int),
        ('descrSize', c_int),
        ('frames', POINTER(VLDsiftKeypoint)),
        ('descrs', c_float_p),

        ('numBinAlloc', c_int),
        ('numFrameAlloc', c_int),
        ('numGradAlloc', c_int),

        ('grads', POINTER(c_float_p)),
        ('convTmp1', c_float_p),
        ('convTmp2', c_float_p),
    ]
VLDsiftFilter_p = POINTER(VLDsiftFilter)

vl_dsift_new = LIB['vl_dsift_new']
vl_dsift_new.restype = VLDsiftFilter_p
vl_dsift_new.argtypes = [c_int, c_int]

vl_dsift_new_basic = LIB['vl_dsift_new_basic']
vl_dsift_new_basic.restype = VLDsiftFilter_p
vl_dsift_new_basic.argtypes = [c_int, c_int, c_int, c_int]

vl_dsift_delete = LIB['vl_dsift_delete']
vl_dsift_delete.restype = None
vl_dsift_delete.argtypes = [VLDsiftFilter_p]

vl_dsift_process = LIB['vl_dsift_process']
vl_dsift_process.restype = None
vl_dsift_process.argtypes = [VLDsiftFilter_p, npc.ndpointer(dtype=np_float)]

_vl_dsift_update_buffers = LIB['_vl_dsift_update_buffers']
_vl_dsift_update_buffers.restype = None
_vl_dsift_update_buffers.argtypes = [VLDsiftFilter_p]

def vl_dsift_set_geometry(self, geom):
    self.geom = geom
    _vl_dsift_update_buffers(self)

def vl_dsift_set_steps(self, step_x, step_y):
    self.stepX = step_x
    self.stepY = step_y
    _vl_dsift_update_buffers(self)

def vl_dsift_set_bounds(self, min_x, min_y, max_x, max_y):
    self.boundMinX = min_x
    self.boundMinY = min_y
    self.boundMaxX = max_x
    self.boundMaxY = max_y
    _vl_dsift_update_buffers(self)

def vl_dsift_set_flat_window(self, use_flat_window):
    self.useFlatWindow = use_flat_window

def vl_dsift_set_window_size(self, window_size):
    assert window_size >= 0.0
    self.windowSize = window_size

def vl_dsift_get_keypoint_num(self):
    return self.numFrames

def vl_dsift_get_descriptor_size(self):
    return self.descrSize

def vl_dsift_get_geometry(self):
    return self.geom

def vl_dsift_get_steps(self):
    return (self.stepX, self.stepY)

def vl_dsift_get_bounds(self):
    return (self.boundMinX, self.boundMinY, self.boundMaxX, self.boundMaxY)

def vl_dsift_get_flat_window(self):
    return self.useFlatWindow

def vl_dsift_get_window_size(self):
    return self.windowSize

def vl_dsift_get_keypoints(self):
    return self.frames

def vl_dsift_get_descriptors(self):
    return self.descrs

def vl_dsift_transpose_descriptor(dest, src, num_bin_t, num_bin_x, num_bin_y):
    for y in xrange(num_bin_y):
        for x in xrange(num_bin_x):
            offset = num_bin_t * (x + y * num_bin_x)
            offsetT = num_bin_t * (y + x * num_bin_x)
            for t in xrange(num_bin_t):
                tT = num_bin_t // 4 - t
                dest[offsetT + (tT + num_bin_t) % num_bin_t] = src[offset + t]

################################################################################

def vl_dsift(data, verbose=False, fast=False, norm=False, bounds=None,
             size=3, step=1, window_size=None, float_descriptors=False,
             geometry=(4, 4, 8)):
    '''
    Dense sift descriptors from an image.

    Returns:
        frames: num_frames x (2 or 3) matrix of x, y, (norm)
        descrs: num_frames x 128 matrix of descriptors
    '''

    data = as_float_image(data, dtype=np.float32, order='F')
    if data.ndim != 2:
        raise TypeError("data should be a 2d array")

    geom = VLDsiftDescriptorGeometry()
    geom.numBinX, geom.numBinY, geom.numBinT = geometry
    geom.binSizeX, geom.binSizeY = (size, size) if np.isscalar(size) else size

    check_integer(geom.binSizeX, "size[0]", 1)
    check_integer(geom.binSizeY, "size[1]", 1)
    check_integer(geom.numBinX, "geometry[0]", 1)
    check_integer(geom.numBinY, "geometry[1]", 1)
    check_integer(geom.numBinT, "geometry[2]", 1)

    step = (step, step) if np.isscalar(step) else step
    check_integer(step[0], "step[0]", 1)
    check_integer(step[1], "step[1]", 1)

    if window_size is not None:
        check_float(window_size, "window_size", 0)

    # construct the dsift object
    M, N = data.shape
    dsift_p = vl_dsift_new(M, N)
    try:
        dsift = dsift_p.contents
        vl_dsift_set_geometry(dsift, geom)
        vl_dsift_set_steps(dsift, step[0], step[1])

        if bounds is not None:
            y0, x0, y1, x1 = bounds
            vl_dsift_set_bounds(dsift, int(max(x0, 0)),
                                       int(max(y0, 0)),
                                       int(min(x1, M - 1)),
                                       int(min(y1, N - 1)))

        vl_dsift_set_flat_window(dsift, fast)

        if window_size is not None:
            vl_dsift_set_window_size(dsift, window_size)

        num_frames = vl_dsift_get_keypoint_num(dsift)
        descr_size = vl_dsift_get_descriptor_size(dsift)
        geom = vl_dsift_get_geometry(dsift)

        if verbose:
            step_x, step_y = vl_dsift_get_steps(dsift)
            min_y, min_x, max_y, max_x = vl_dsift_get_bounds(dsift)
            use_flat_window = vl_dsift_get_flat_window(dsift)

            print("vl_dsift: image size         [W, H] = [{}, {}]".format(N, M))
            print("vl_dsift: bounds:            [minX,minY,maxX,maxY] = [{}, {}, {}, {}]"
                .format(min_x + 1, min_y + 1, max_x + 1, max_y + 1))
            print("vl_dsift: subsampling steps: stepX={}, stepY={}".format(step_x, step_y))
            print("vl_dsift: num bins:          [numBinT, numBinX, numBinY] = [{}, {}, {}]"
                .format(geom.numBinT, geom.numBinX, geom.numBinY))
            print("vl_dsift: descriptor size:   {}".format(descr_size))
            print("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [{}, {}]"
                .format(geom.binSizeX, geom.binSizeY))
            print("vl_dsift: flat window:       {}".format(bool(use_flat_window)))
            print("vl_dsift: window size:       {}".format(vl_dsift_get_window_size(dsift)))
            print("vl_dsift: num of features:   {}".format(num_frames))

        vl_dsift_process(dsift_p, data)

        # copy frames' locations, norms out
        # the frames are a structure of just 4 doubles (VLDsiftKeypoint)
        frames = cast(vl_dsift_get_keypoints(dsift), c_double_p)
        frames = npc.as_array(frames, shape=(num_frames, 4))
        cols = [1, 0] # y, x
        if norm:
            cols.append(3) # norm
        frames = np.require(frames[:, cols], requirements=['C', 'O'])
        # NOTE: In contrast to Matlab, the assignment indices start with 0

        # copy descriptors into a new array
        descrs = npc.as_array(vl_dsift_get_descriptors(dsift), shape=(num_frames, descr_size))
        descrs = np.require(descrs * 512, requirements='O')
        np.minimum(descrs, 255, out=descrs)
        if not float_descriptors:
            descrs = descrs.astype(np.uint8)  # TODO: smarter about copying?
        new_order = np.empty(descr_size, dtype=int)
        vl_dsift_transpose_descriptor(new_order, np.arange(descr_size),
            geom.numBinT, geom.numBinX, geom.numBinY)
        descrs = descrs[:, new_order]

        return frames, descrs

    finally:
        vl_dsift_delete(dsift_p)
