from __future__ import print_function

from collections import namedtuple
from ctypes import cast, POINTER, Structure, c_int, c_float, c_double, c_void_p

import numpy as np
import numpy.ctypeslib as npc

from .vl_ctypes import (load_library, Enum, vl_size, vl_type,
                        np_to_c_types, c_to_vl_types, c_to_np_types)
from .utils import check_integer

LIB = load_library()

np_double = c_to_np_types[c_double]

class GmmInitialization(Enum):
    KMEANS = 0
    RAND = 1
    # CUSTOM = 2

class VlGMM(Structure):
    _fields_ = [
        ('_opaque', c_int),
    ]
VlGMM_p = POINTER(VlGMM)

vl_gmm_new = LIB['vl_gmm_new']
vl_gmm_new.restype = VlGMM_p
vl_gmm_new.argtypes = [vl_type, vl_size, vl_size]

vl_gmm_new_copy = LIB['vl_gmm_new_copy']
vl_gmm_new_copy.restype = VlGMM_p
vl_gmm_new_copy.argtypes = [VlGMM_p]

vl_gmm_delete = LIB['vl_gmm_delete']
vl_gmm_delete.restype = None
vl_gmm_delete.argtypes = [VlGMM_p]

vl_gmm_reset = LIB['vl_gmm_reset']
vl_gmm_reset.restype = None
vl_gmm_reset.argtypes = [VlGMM_p]

vl_gmm_set_verbosity = LIB['vl_gmm_set_verbosity']
vl_gmm_set_verbosity.restype = None
vl_gmm_set_verbosity.argtypes = [VlGMM_p, c_int]

vl_gmm_set_num_repetitions = LIB['vl_gmm_set_num_repetitions']
vl_gmm_set_num_repetitions.restype = None
vl_gmm_set_num_repetitions.argtypes = [VlGMM_p, vl_size]

vl_gmm_set_max_num_iterations = LIB['vl_gmm_set_max_num_iterations']
vl_gmm_set_max_num_iterations.restype = None
vl_gmm_set_max_num_iterations.argtypes = [VlGMM_p, vl_size]

vl_gmm_set_initialization = LIB['vl_gmm_set_initialization']
vl_gmm_set_initialization.restype = None
vl_gmm_set_initialization.argtypes = [VlGMM_p, GmmInitialization]

vl_gmm_set_covariance_lower_bound = LIB['vl_gmm_set_covariance_lower_bound']
vl_gmm_set_covariance_lower_bound.restype = None
vl_gmm_set_covariance_lower_bound.argtypes = [VlGMM_p, c_double]

vl_gmm_set_covariance_lower_bounds = LIB['vl_gmm_set_covariance_lower_bounds']
vl_gmm_set_covariance_lower_bounds.restype = None
vl_gmm_set_covariance_lower_bounds.argtypes = [VlGMM_p, POINTER(c_double)]

vl_gmm_get_initialization = LIB['vl_gmm_get_initialization']
vl_gmm_get_initialization.restype = GmmInitialization
vl_gmm_get_initialization.argtypes = [VlGMM_p]

vl_gmm_get_max_num_iterations = LIB['vl_gmm_get_max_num_iterations']
vl_gmm_get_max_num_iterations.restype = vl_size
vl_gmm_get_max_num_iterations.argtypes = [VlGMM_p]

vl_gmm_get_num_repetitions = LIB['vl_gmm_get_num_repetitions']
vl_gmm_get_num_repetitions.restype = vl_size
vl_gmm_get_num_repetitions.argtypes = [VlGMM_p]

vl_gmm_get_data_type = LIB['vl_gmm_get_data_type']
vl_gmm_get_data_type.restype = vl_type
vl_gmm_get_data_type.argtypes = [VlGMM_p]

vl_gmm_get_covariance_lower_bounds = LIB['vl_gmm_get_covariance_lower_bounds']
vl_gmm_get_covariance_lower_bounds.restype = POINTER(c_double)
vl_gmm_get_covariance_lower_bounds.argtypes = [VlGMM_p]

vl_gmm_cluster = LIB['vl_gmm_cluster']
vl_gmm_cluster.restype = c_double
vl_gmm_cluster.argtypes = [VlGMM_p, c_void_p, vl_size]

vl_gmm_get_num_clusters = LIB['vl_gmm_get_num_clusters']
vl_gmm_get_num_clusters.restype = vl_size
vl_gmm_get_num_clusters.argtypes = [VlGMM_p]

vl_gmm_get_means = LIB['vl_gmm_get_means']
vl_gmm_get_means.restype = c_void_p
vl_gmm_get_means.argtypes = [VlGMM_p]

vl_gmm_get_covariances = LIB['vl_gmm_get_covariances']
vl_gmm_get_covariances.restype = c_void_p
vl_gmm_get_covariances.argtypes = [VlGMM_p]

vl_gmm_get_priors = LIB['vl_gmm_get_priors']
vl_gmm_get_priors.restype = c_void_p
vl_gmm_get_priors.argtypes = [VlGMM_p]

vl_gmm_get_posteriors = LIB['vl_gmm_get_posteriors']
vl_gmm_get_posteriors.restype = c_void_p
vl_gmm_get_posteriors.argtypes = [VlGMM_p]

################################################################################

# FIXME: init priors/means/coveriances not supported yet
def vl_gmm(data, num_clusters, ret_loglikelihood=False, ret_posterior=False,
           verbose=False, max_iter=100, cov_bound=None, initialization='rand',
           num_rep=1):

    # NOTE: We expect one entry per row instead of column
    data = np.asarray(data, order='C')
    c_dtype = np_to_c_types.get(data.dtype, None)
    if c_dtype not in [c_float, c_double]:
        raise TypeError("data should be float32 or float64")
    vl_dtype = c_to_vl_types[c_dtype]

    if data.ndim != 2:
        raise TypeError("data should be a 2d array")
    num_data, dim = data.shape
    if dim == 0:
        raise ValueError("data dimension is zero")
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains INFs or NaNs")

    if cov_bound is not None and not np.isscalar(cov_bound):
        cov_bound = np.asarray(cov_bound, dtype=np_double)
        if cov_bound.shape != (dim,):
            raise ValueError("cov_bound does not have the correct size")

    check_integer(num_clusters, "num_clusters", 1, num_data)
    check_integer(max_iter, "max_iter", 0)
    check_integer(num_rep, "num_rep", 1)

    initialization = GmmInitialization._members[initialization.upper()]

    gmm_p = vl_gmm_new(vl_dtype, dim, num_clusters)
    try:
        vl_gmm_set_verbosity(gmm_p, verbose)
        vl_gmm_set_num_repetitions(gmm_p, num_rep)
        vl_gmm_set_max_num_iterations(gmm_p, max_iter)
        vl_gmm_set_initialization(gmm_p, initialization)
        if cov_bound is not None:
            if np.isscalar(cov_bound):
                vl_gmm_set_covariance_lower_bound(gmm_p, cov_bound)
            else:
                cov_bound_p = data.ctypes.data_as(c_void_p)
                vl_gmm_set_covariance_lower_bounds(gmm_p, cov_bound_p)
        # vl_gmm_set_priors(gmm_p, init_priors)
        # vl_gmm_set_means(gmm_p, init_means)
        # vl_gmm_set_covariances(gmm_p, init_covariances)

        if verbose:
            initializationName = vl_gmm_get_initialization(gmm_p).name
            cov_lower_bounds = vl_gmm_get_covariance_lower_bounds(gmm_p)
            cov_lower_bounds = npc.as_array(cov_lower_bounds, (dim,))

            print("vl_gmm: initialization = {}".format(initializationName))
            print("vl_gmm: maxNumIterations = {}".format(vl_gmm_get_max_num_iterations(gmm_p)))
            print("vl_gmm: numRepetitions = {}".format(vl_gmm_get_num_repetitions(gmm_p)))
            print("vl_gmm: data type = {}".format(vl_gmm_get_data_type(gmm_p).name))
            print("vl_gmm: data dimension = {}".format(dim))
            print("vl_gmm: num. data points = {}".format(num_data))
            print("vl_gmm: num. Gaussian modes = {}".format(num_clusters))
            print("vl_gmm: lower bound on covariance = {}".format(cov_lower_bounds))

        data_p = data.ctypes.data_as(c_void_p)
        ll = vl_gmm_cluster(gmm_p, data_p, num_data)

        # copy centers
        means = cast(vl_gmm_get_means(gmm_p), POINTER(c_dtype))
        means = npc.as_array(means, (vl_gmm_get_num_clusters(gmm_p), dim))
        means = np.require(means, requirements='O')

        covariances = cast(vl_gmm_get_covariances(gmm_p), POINTER(c_dtype))
        covariances = npc.as_array(covariances, (vl_gmm_get_num_clusters(gmm_p), dim))
        covariances = np.require(covariances, requirements='O')

        priors = cast(vl_gmm_get_priors(gmm_p), POINTER(c_dtype))
        priors = npc.as_array(priors, (vl_gmm_get_num_clusters(gmm_p),))
        priors = np.require(priors, requirements='O')

        ret = [means, covariances, priors]
        ret_fields = ['means', 'covariances', 'priors']

        # optionally return loglikelihood
        if ret_loglikelihood:
            ret.append(ll)
            ret_fields.append('ll')

        # optionally return posterior probabilities
        if ret_posterior:
            posteriors = cast(vl_gmm_get_posteriors(gmm_p), POINTER(c_dtype))
            posteriors = npc.as_array(posteriors, (num_data, vl_gmm_get_num_clusters(gmm_p)))
            posteriors = np.require(posteriors, requirements='O')

            ret.append(posteriors)
            ret_fields.append('posteriors')

        return namedtuple('GmmRetVal', ret_fields)(*ret)

    finally:
        vl_gmm_delete(gmm_p)
