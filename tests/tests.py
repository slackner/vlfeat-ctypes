#!/usr/bin/env python2
from __future__ import division

from PIL import Image
import math
import numpy
import numpy.random
import os
import sys
import unittest

import_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(1, import_path)

import vlfeat
from vlfeat.utils import rgb2gray

class Tests(unittest.TestCase):
    def test_all(self):
        img = numpy.array(Image.open('roofs1.jpg'))

        # Test rgb2gray
        img_gray = rgb2gray(img)
        self.assertEqual(tuple(img_gray.shape), (478, 640))
        self.assertTrue(numpy.allclose(img_gray[:4, :4], numpy.array([[ 0.8973, 0.8973, 0.8973, 0.9052 ],
                                                                      [ 0.8973, 0.8973, 0.8973, 0.9052 ],
                                                                      [ 0.8973, 0.8973, 0.9021, 0.9061 ],
                                                                      [ 0.9013, 0.9013, 0.9061, 0.9100 ]]), atol=1e-4))
        if os.path.exists("img_gray.txt"):
            expected = numpy.loadtxt("img_gray.txt", delimiter='\t')
            self.assertTrue(numpy.allclose(img_gray, expected, atol=1e-4))

        # Test vl_imsmooth
        binsize = 8
        magnif  = 3
        img_smooth = vlfeat.vl_imsmooth(img_gray, math.sqrt((binsize / magnif)**2 - 0.25), verbose=True)
        self.assertEqual(tuple(img_smooth.shape), (478, 640))
        self.assertTrue(numpy.allclose(img_smooth[:4, :4], numpy.array([[ 0.8998, 0.9013, 0.9034, 0.9057 ],
                                                                        [ 0.9000, 0.9015, 0.9035, 0.9057 ],
                                                                        [ 0.9002, 0.9017, 0.9036, 0.9057 ],
                                                                        [ 0.9005, 0.9020, 0.9038, 0.9058 ]]), atol=1e-4))
        if os.path.exists("img_smooth.txt"):
            expected = numpy.loadtxt("img_smooth.txt", delimiter='\t')
            self.assertTrue(numpy.allclose(img_smooth, expected, atol=1e-4))

        # Test vl_dsift
        frames, descrs = vlfeat.vl_dsift(img_smooth, size=binsize, verbose=True)
        frames = numpy.add(frames.transpose(), 1)
        descrs = descrs.transpose()
        self.assertEqual(tuple(frames.shape), (2, 279664))
        self.assertEqual(tuple(descrs.shape), (128, 279664))
        self.assertTrue(numpy.allclose(frames[:, :4], numpy.array([[13, 13, 13, 13], [13, 14, 15, 16]])))
        self.assertTrue(numpy.allclose(descrs[:, 0], numpy.array([134,  35,   0,   0,   0,   0,   0,   5, 109,   9,   1,   0,   0,   0,   0,  61,
                                                                    7,   2,  32,  21,   0,   0,   1,  28,   2,  13, 111,   9,   0,   0,   0,   2,
                                                                   33, 134, 131,   0,   0,   0,   0,   0,  30,  92, 134,   0,   0,   0,   0,  19,
                                                                   11,  42, 134,   0,   0,   0,   1,  31,   6,  20, 124,   3,   0,   0,   0,   7,
                                                                    5, 134, 134,   0,   0,   0,   0,   0,   1,  94, 134,   0,   0,   0,   0,   0,
                                                                    0,  34, 134,   1,   0,   0,   0,   0,   0,   4, 134,  13,   0,   2,   2,   0,
                                                                   27,  53,  15,   0,   0,   0,   0,   1,  11,  48,  27,   2,   0,   0,   0,   0,
                                                                    0,   5,  28,  16,   1,   0,   0,   0,   0,   2,  13,  16,   4,   5,   4,   0])))
        if os.path.exists("dsift_frames.txt"):
            expected = numpy.loadtxt("dsift_frames.txt", delimiter='\t')
            self.assertTrue(numpy.allclose(frames, expected))
        if os.path.exists("dsift_descrs.txt"):
            expected = numpy.loadtxt("dsift_descrs.txt", delimiter='\t')
            self.assertTrue(numpy.linalg.norm(expected - descrs) < 28) # rounding errors?

        # Test vl_kmeans
        centers, assigns = vlfeat.vl_kmeans(numpy.array([[1], [2], [3], [10], [11], [12]], dtype='f'), 2, ret_quantize=True, verbose=True)
        self.assertTrue(numpy.allclose(centers, numpy.array([[2], [11]])))
        self.assertTrue(numpy.allclose(assigns, numpy.array([0, 0, 0, 1, 1, 1])))

        centers, assigns = vlfeat.vl_kmeans(numpy.array([[1, 0], [2, 0], [3, 0], [10, 1], [11, 1], [12, 1]], dtype='f'), 2, ret_quantize=True)
        self.assertTrue(numpy.allclose(centers, numpy.array([[11, 1], [2, 0]]))) # order swapped?
        self.assertTrue(numpy.allclose(assigns, numpy.array([1, 1, 1, 0, 0, 0])))

        # Test vl_gmm
        if os.path.exists("gmm_data.txt"):
            data = numpy.loadtxt("gmm_data.txt", delimiter='\t').transpose()
        else:
            data = numpy.random.rand(5000, 2)

        means, covariances, priors, ll, posteriors = vlfeat.vl_gmm(data, 30, verbose=True, ret_loglikelihood=True, ret_posterior=True)
        self.assertEqual(tuple(means.shape), (30, 2))
        self.assertEqual(tuple(covariances.shape), (30, 2))
        self.assertEqual(tuple(priors.shape), (30,))
        self.assertEqual(tuple(posteriors.shape), (5000, 30))

        if os.path.exists("gmm_means.txt"):
            expected = numpy.loadtxt("gmm_means.txt", delimiter='\t').transpose()
            self.assertTrue(numpy.allclose(means, expected, atol=1e-4))

        if os.path.exists("gmm_covariances.txt"):
            expected = numpy.loadtxt("gmm_covariances.txt", delimiter='\t').transpose()
            self.assertTrue(numpy.allclose(covariances, expected, atol=1e-4))

        if os.path.exists("gmm_priors.txt"):
            expected = numpy.loadtxt("gmm_priors.txt", delimiter='\t').transpose()
            self.assertTrue(numpy.allclose(priors, expected, atol=1e-4))

        if os.path.exists("gmm_posteriors.txt"):
            expected = numpy.loadtxt("gmm_posteriors.txt", delimiter='\t').transpose()
            self.assertTrue(numpy.allclose(posteriors, expected, atol=1e-3))

        # Test vl_fisher
        if os.path.exists("fisher_data.txt"):
            data = numpy.loadtxt("fisher_data.txt", delimiter='\t').transpose()
        else:
            data = numpy.random.rand(1000, 2)

        encoding = vlfeat.vl_fisher(data, means, covariances, priors, verbose=True)
        self.assertEqual(tuple(encoding.shape), (120,))

        if os.path.exists("fisher_encoding.txt"):
            expected = numpy.loadtxt("fisher_encoding.txt", delimiter='\t').transpose()
            self.assertTrue(numpy.allclose(encoding, expected, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
