# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:48:27 2020

@author: JG07DA
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt


size = 200
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
                if np.abs(z) <= 2:
                    z = z * z + c
                    m[i, j] = n
                else:
                    break
    return m

#s = timeit.timeit(mandelbrot_python(size, iterations))
#print(s)
m = mandelbrot_python(size, iterations)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.xmin = -2.0
plt.xmax= 1.0
plt.ymin= -1.5
plt.ymax= 1.5
ax.imshow(np.log(m), cmap=plt.cm.hot)
ax.set_axis_on()

