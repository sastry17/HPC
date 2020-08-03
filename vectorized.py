# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:40:53 2020

@author: JG07DA
"""

from numba import jit, vectorize, complex64, int32
import numpy as np

@vectorize([int32(complex64, int32)], target='parallel')
def mandel_numba_vect(c, maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return n
        
def mandel_set_numba_vect(xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n = mandel_numba_vect(c,maxiter)
    return n

