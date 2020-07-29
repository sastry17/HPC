# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:43:21 2020

@author: JG07DA
"""

import unittest
import variant
import numpy as np


class TestMandelBrot(unittest.TestCase):
    
    def testMethods(self):
        self.methods = ['variant.mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)']
    
    
    def test_mandelbrot_naive(self):
        