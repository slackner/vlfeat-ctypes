from __future__ import division, print_function
import numpy as np
from ctypes import c_int, c_float, c_double, c_void_p
from .vl_ctypes import load_library, vl_size, vl_type, np_to_c_types, c_to_vl_types

LIB = load_library()

VL_FISHER_FLAG_SQUARE_ROOT = 1 << 0
VL_FISHER_FLAG_NORMALIZED  = 1 << 1
VL_FISHER_FLAG_IMPROVED    = VL_FISHER_FLAG_NORMALIZED | VL_FISHER_FLAG_SQUARE_ROOT
VL_FISHER_FLAG_FAST        = 1 << 2

vl_fisher_encode = LIB['vl_fisher_encode']
vl_fisher_encode.restype = vl_size
vl_fisher_encode.argtypes = [c_void_p, vl_type,             # enc, dataType
                             c_void_p, vl_size, vl_size,    # means, dimension, numClusters
                             c_void_p,                      # coveriances
                             c_void_p,                      # priors
                             c_void_p, vl_size,             # data, numData
                             c_int]                         # flags

################################################################################

def vl_fisher(data, means, covariances, priors, verbose=False,
              normalized=False, square_root=False, improved=False, fast=False):

    # NOTE: We expect one entry per row instead of column
    data = np.asarray(data, order='C')
    means = np.asarray(means, order='C')
    covariances = np.asarray(covariances, order='C')
    priors = np.asarray(priors, order='C')

    c_dtype = np_to_c_types.get(data.dtype, None)
    if c_dtype not in [c_float, c_double]:
        raise TypeError("data should be float32 or float64")
    if np_to_c_types.get(means.dtype, None) != c_dtype:
        raise TypeError("means should have the same type as data")
    if np_to_c_types.get(covariances.dtype, None) != c_dtype:
        raise TypeError("covariances should have the same type as data")
    if np_to_c_types.get(priors.dtype, None) != c_dtype:
        raise TypeError("priors should have the same type as data")
    vl_dtype = c_to_vl_types[c_dtype]

    if data.ndim != 2:
        raise TypeError("data should be a 2d array")
    num_data, dim = data.shape
    num_clusters = means.shape[0]
    if dim == 0:
        raise ValueError("data dimension is zero")
    if means.shape != (num_clusters, dim):
        raise ValueError("means does not have the correct size")
    if covariances.shape != (num_clusters, dim):
        raise ValueError("covariances does not have the correct size")
    if priors.shape != (num_clusters,):
        raise ValueError("priors does not have the correct size")

    flags = 0
    if normalized:
        flags = flags | VL_FISHER_FLAG_NORMALIZED
    if square_root:
        flags = flags | VL_FISHER_FLAG_SQUARE_ROOT
    if improved:
        flags = flags | VL_FISHER_FLAG_IMPROVED
    if fast:
        flags = flags | VL_FISHER_FLAG_FAST

    if verbose:
        normalized = (flags & VL_FISHER_FLAG_NORMALIZED) != 0
        square_root = (flags & VL_FISHER_FLAG_SQUARE_ROOT) != 0
        fast = (flags & VL_FISHER_FLAG_FAST) != 0

        print("vl_fisher: num data: {}".format(num_data))
        print("vl_fisher: num clusters: {}".format(num_clusters))
        print("vl_fisher: data dimension: {}".format(dim))
        print("vl_fisher: code dimension: {}".format(num_clusters * dim))
        print("vl_fisher: square root: {}".format(square_root))
        print("vl_fisher: normalized: {}".format(normalized))
        print("vl_fisher: fast: {}".format(fast))

    output = np.empty((dim * num_clusters * 2,), dtype=data.dtype)

    output_p = output.ctypes.data_as(c_void_p)
    data_p = data.ctypes.data_as(c_void_p)
    means_p = means.ctypes.data_as(c_void_p)
    covariances_p = covariances.ctypes.data_as(c_void_p)
    priors_p = priors.ctypes.data_as(c_void_p)

    num_terms = vl_fisher_encode(output_p, vl_dtype, means_p, dim, num_clusters,
                                 covariances_p, priors_p, data_p, num_data, flags)

    if verbose:
        print("vl_fisher: sparsity of assignments: {}% ({} non-negligible assignments)"
                .format(100.0 * (1 - num_terms / (num_data * num_clusters + 1e-12)), num_terms))

    return output
