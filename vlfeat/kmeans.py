from __future__ import print_function

from collections import namedtuple
from ctypes import (c_int, c_float, c_double, c_void_p,
                    POINTER, CFUNCTYPE, cast)

import numpy as np
import numpy.ctypeslib as npc

from .vl_ctypes import (load_library, CustomStructure, Enum,
                        vl_type, vl_size, np_to_c_types, c_to_vl_types)
from .utils import check_integer, check_float

LIB = load_library()

class VectorComparisonType(Enum):
    DISTANCE_L1 = 0
    DISTANCE_L2 = 1
    DISTANCE_CHI2 = 2
    DISTANCE_HELLINGER = 3
    DISTANCE_JS = 4
    DISTANCE_MAHALANOBIS = 5
    KERNEL_L1 = 6
    KERNEL_L2 = 7
    KERNEL_CHI2 = 8
    KERNEL_HELLINGER = 9
    KERNEL_JS = 10

FloatVectorComparisonFunction = POINTER(CFUNCTYPE(
    c_float, vl_size, POINTER(c_float), POINTER(c_float)))
DoubleVectorComparisonFunction = POINTER(CFUNCTYPE(
    c_double, vl_size, POINTER(c_double), POINTER(c_double)))

class KMeansAlgorithm(Enum):
    LLOYD = 0
    ELKAN = 1
    ANN = 2

class KMeansInitialization(Enum):
    RANDOM = 0
    PLUSPLUS = 1

class VlKMeans(CustomStructure):
    _fields_ = [
        ('dataType', vl_type),
        ('dimension', vl_size),
        ('numCenters', vl_size),
        ('numTrees', vl_size),
        ('maxNumComparisons', vl_size),

        ('initialization', KMeansInitialization),
        ('algorithm', KMeansAlgorithm),
        ('distance', VectorComparisonType),
        ('maxNumIterations', vl_size),
        ('minEnergyVariation', c_double),
        ('numRepetitions', vl_size),
        ('verbosity', c_int),

        ('centers', c_void_p),
        ('centerDistances', c_void_p),

        ('energy', c_double),
        ('floatVectorComparisonFn', FloatVectorComparisonFunction),
        ('doubleVectorComparisonFn', DoubleVectorComparisonFunction),
    ]
VlKMeans_p = POINTER(VlKMeans)

vl_kmeans_new = LIB['vl_kmeans_new']
vl_kmeans_new.restype = VlKMeans_p
vl_kmeans_new.argtypes = [vl_type, VectorComparisonType]

vl_kmeans_new_copy = LIB['vl_kmeans_new_copy']
vl_kmeans_new_copy.restype = VlKMeans_p
vl_kmeans_new_copy.argtypes = [VlKMeans_p]

vl_kmeans_delete = LIB['vl_kmeans_delete']
vl_kmeans_delete.restype = None
vl_kmeans_delete.argtypes = [VlKMeans_p]

vl_kmeans_reset = LIB['vl_kmeans_reset']
vl_kmeans_reset.restype = None
vl_kmeans_reset.argtypes = [VlKMeans_p]

vl_kmeans_cluster = LIB['vl_kmeans_cluster']
vl_kmeans_cluster.restype = c_double
vl_kmeans_cluster.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_quantize = LIB['vl_kmeans_quantize']
vl_kmeans_quantize.restype = None
vl_kmeans_quantize.argtypes = [VlKMeans_p, npc.ndpointer(dtype=np.uint32), c_void_p, c_void_p, vl_size]

vl_kmeans_set_centers = LIB['vl_kmeans_set_centers']
vl_kmeans_set_centers.restype = None
vl_kmeans_set_centers.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size]

vl_kmeans_init_centers_with_rand_data = LIB['vl_kmeans_init_centers_with_rand_data']
vl_kmeans_init_centers_with_rand_data.restype = None
vl_kmeans_init_centers_with_rand_data.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_init_centers_plus_plus = LIB['vl_kmeans_init_centers_plus_plus']
vl_kmeans_init_centers_plus_plus.restype = None
vl_kmeans_init_centers_plus_plus.argtypes = [VlKMeans_p, c_void_p, vl_size, vl_size, vl_size]

vl_kmeans_refine_centers = LIB['vl_kmeans_refine_centers']
vl_kmeans_refine_centers.restype = c_double
vl_kmeans_refine_centers.argtypes = [VlKMeans_p, c_void_p, vl_size]

def vl_kmeans_set_verbosity(self, verbosity):
    self.verbosity = verbosity

def vl_kmeans_set_num_repetitions(self, num_rep):
    assert num_rep >= 1
    self.numRepetitions = num_rep

def vl_kmeans_get_num_repetitions(self):
    return self.numRepetitions

def vl_kmeans_set_algorithm(self, algorithm):
    self.algorithm = algorithm

def vl_kmeans_get_algorithm(self):
    return self.algorithm

def vl_kmeans_set_initialization(self, initialization):
    self.initialization = initialization

def vl_kmeans_get_initialization(self):
    return self.initialization

def vl_kmeans_set_max_num_iterations(self, max_iter):
    self.maxNumIterations = max_iter

def vl_kmeans_get_max_num_iterations(self):
    return self.maxNumIterations

def vl_kmeans_set_max_num_comparisons(self, max_compare):
    self.maxNumComparisons = max_compare

def vl_kmeans_set_num_trees(self, num_trees):
    self.numTrees = num_trees

def vl_kmeans_set_min_energy_variation(self, min_energy_var):
    assert min_energy_var >= 0
    self.minEnergyVariation = min_energy_var

def vl_kmeans_get_min_energy_variation(self):
    return self.minEnergyVariation

def vl_kmeans_get_data_type(self):
    return self.dataType

def vk_kmeans_get_distance(self):
    return self.distance

def vl_kmeans_get_centers(self):
    return self.centers

def vl_kmeans_get_num_centers(self):
    return self.numCenters

################################################################################

def vl_kmeans(data, num_centers, ret_quantize=False, ret_energy=False,
              verbose=False, max_iter=100, min_energy_var=None,
              algorithm='lloyd', initialization='plusplus', distance='l2',
              num_rep=1, num_trees=3, max_compare=100):

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

    check_integer(num_centers, "num_centers", 1, num_data)
    check_integer(max_iter, "max_iter", 0)
    if min_energy_var is not None:
        check_float(min_energy_var, "min_energy_var", 0)
    check_integer(num_rep, "num_rep", 1)
    check_integer(num_trees, "num_trees", 1)
    check_integer(max_compare, "max_compare", 0)

    algorithm = KMeansAlgorithm._members[algorithm.upper()]
    initialization = KMeansInitialization._members[initialization.upper()]
    distance = VectorComparisonType._members['DISTANCE_' + distance.upper()]

    kmeans_p = vl_kmeans_new(vl_dtype, distance)
    try:
        kmeans = kmeans_p.contents
        vl_kmeans_set_verbosity(kmeans, 1 if verbose else 0)
        vl_kmeans_set_num_repetitions(kmeans, num_rep)
        vl_kmeans_set_algorithm(kmeans, algorithm)
        vl_kmeans_set_initialization(kmeans, initialization)
        vl_kmeans_set_max_num_iterations(kmeans, max_iter)
        vl_kmeans_set_max_num_comparisons(kmeans, max_compare)
        vl_kmeans_set_num_trees(kmeans, num_trees)
        if min_energy_var is not None:
            vl_kmeans_set_min_energy_variation(kmeans, min_energy_var)

        if verbose:
            algorithmName = vl_kmeans_get_algorithm(kmeans).name
            initializationName = vl_kmeans_get_initialization(kmeans).name

            print("kmeans: Initialization = {}".format(initializationName))
            print("kmeans: Algorithm = {}".format(algorithmName))
            print("kmeans: MaxNumIterations = {}".format(vl_kmeans_get_max_num_iterations(kmeans)))
            print("kmeans: minEnergyVariation = {}".format(vl_kmeans_get_min_energy_variation(kmeans)))
            print("kmeans: NumRepetitions = {}".format(vl_kmeans_get_num_repetitions(kmeans)))
            print("kmeans: data type = {}".format(vl_kmeans_get_data_type(kmeans).name))
            print("kmeans: distance = {}".format(vk_kmeans_get_distance(kmeans).name))
            print("kmeans: data dimension = {}".format(dim))
            print("kmeans: num. data points = {}".format(num_data))
            print("kmeans: num. centers = {}".format(num_centers))
            print("kmeans: max num. comparisons = {}".format(max_compare))
            print("kmeans: num. trees = {}".format(num_trees))
            print()

        data_p = data.ctypes.data_as(c_void_p)
        energy = vl_kmeans_cluster(kmeans_p, data_p, dim, num_data, num_centers)

        # copy centers
        centers = cast(vl_kmeans_get_centers(kmeans), POINTER(c_dtype))
        centers = npc.as_array(centers, (vl_kmeans_get_num_centers(kmeans), dim))
        centers = np.require(centers, requirements='O')

        if not ret_quantize and not ret_energy:
            return centers

        ret = [centers]
        ret_fields = ['centers']

        # optionally quantize
        if ret_quantize:
            assignments = np.empty(num_data, dtype=np.uint32)
            vl_kmeans_quantize(kmeans_p, assignments, None, data_p, num_data)
            # NOTE: In contrast to Matlab, the assignment indices start with 0

            ret.append(assignments)
            ret_fields.append('assignments')

        # optionally return energy
        if ret_energy:
            ret.append(energy)
            ret_fields.append('energy')

        return namedtuple('KMeansRetVal', ret_fields)(*ret)

    finally:
        vl_kmeans_delete(kmeans_p)
