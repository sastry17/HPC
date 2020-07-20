# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:48:27 2020

@author: JG07DA
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
size = 400
iterations = 100

@jit
def mandelbrot_python(size, iterations):
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            c = (-2 + 3. / size * j +
                 1j * (1.5 - 3. / size * i))
            z = 0
            for n in range(iterations):
                if np.abs(z) <= 10:
                    z = z * z + c
                    m[i, j] = n
                else:
                    break
    return m

#s = timeit.timeit(mandelbrot_python(size, iterations))
#print(s)
m = mandelbrot_python(size, iterations)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(np.log(m), cmap=plt.cm.hot)
ax.set_axis_off()

