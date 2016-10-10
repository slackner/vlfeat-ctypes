#!/usr/bin/env python2
from __future__ import division

from PIL import Image
import math
import numpy
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
        expected = numpy.loadtxt("img_gray.txt", delimiter='\t')
        self.assertTrue(numpy.allclose(img_gray, expected, atol=1e-4))

        # Test vl_imsmooth
        binsize = 8
        magnif  = 3
        img_smooth = vlfeat.vl_imsmooth(img_gray, math.sqrt((binsize / magnif)**2 - 0.25))
        expected = numpy.loadtxt("img_smooth.txt", delimiter='\t')
        self.assertTrue(numpy.allclose(img_smooth, expected, atol=1e-4))

        # Test vl_dsift
        frames, descrs = vlfeat.vl_dsift(img_smooth, size=binsize)
        frames = numpy.add(frames.transpose(), 1)
        descrs = descrs.transpose()
        expected = numpy.loadtxt("dsift_frames.txt", delimiter='\t')
        self.assertTrue(numpy.allclose(frames, expected))
        expected = numpy.loadtxt("dsift_descrs.txt", delimiter='\t')
        self.assertEqual(descrs.shape, expected.shape)
        self.assertTrue(numpy.linalg.norm(expected - descrs) < 28) # rounding errors?

if __name__ == '__main__':
    unittest.main()
