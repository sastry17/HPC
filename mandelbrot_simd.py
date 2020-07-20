# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:20:12 2020

@author: JG07DA
"""


import numpy as np
import matplotlib.pyplot as plt


size = 5000
iterations = 100


# manual initialization for numpy array
def initialize(size): 
    x, y = np.meshgrid(np.linspace(-2, 1, size),
                       np.linspace(-1.5, 1.5, size))
    c = x + 1j * y
    z = c.copy()
    m = np.zeros((size, size))
    return c, z, m

def mandelbrot_simd(c, z, m, iterations):
    for n in range(iterations):
        indices = np.abs(z) <= 10
        z[indices] = z[indices] ** 2 + c[indices]
        m[indices] = n
    return m
        
c, z, m = initialize(size)
m = mandelbrot_simd(c, z, m, iterations)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(np.log(m), cmap=plt.cm.hot)
ax.set_axis_on()

