# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:40:15 2020

@author: JG07DA
"""

from timeit import default_timer as timer
import numpy as np
import time
import time
from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import matplotlib.pyplot as plt
import math
import multiprocessing as mp

def mandelbrot_mp(c, max_iterations=100):
    z = 0
    for i in range(max_iterations):
        z = z**2 + c
        if z.real**2 + z.imag**2 > 4:
            return i+1
        if i == max_iterations-1:
            return 0

def mandelbrot_naive(z,iterations):
    c = z
    for n in range(iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return iterations

def mandelbrot_set_naive(width,height,min_x,max_x,min_y,max_y,maxIterations):
    r1 = np.linspace(min_x, max_x, width)
    r2 = np.linspace(min_y, max_y, height)
    return (r1,r2,[mandelbrot_naive(complex(r, i),maxIterations) for r in r1 for i in r2])


@jit
def mandelbrot_numba(x, y, max_iters):

    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal_numba(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandelbrot_numba(real, imag, iters)
            image[y, x] = color

    return image



@jit(int32(complex64, int32))
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i],maxiter)
        
def mandelbrot_set(width,height,xmin,xmax,ymin,ymax,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
    return (r1,r2,n3.T)



def mandelbrot_dynamic(xmin, xmax, ymin, ymax, N=100):
    incx = math.fabs((xmax - xmin) / N)
    incy = math.fabs((ymax - ymin) / N)
    x, y, myList, lst, n = xmin, ymax, [], [], 1
    while y > ymin and n <= N:
        while x < xmax:
            lst.append(complex(x, y))
            x += incx
        x = xmin
        y -= incy
        myList.extend(lst[:N])
        n += 1
        lst = []

    with mp.Pool() as pool:
        result = pool.map(mandelbrot_mp, myList)

    return np.reshape(np.array(result), (N,N))





def main():
    pre=1000
    pim=1000
    xmin= -2.0
    xmax= 1.0
    ymin= -1.5
    ymax= 1.5
    iterations=200
    image = np.zeros((500 * 2, 500 * 2), dtype=np.uint8)
    
    # ##################################################################################
    start_time = time.time()
    mandelbrot_set_naive(1000,1000,-2.0,0.5,-1.25,1.25,100)
    print("Mandelbrot Naive --- %s seconds ---" % (time.time() - start_time))
    
    # # ##################################################################################
    print("Mandelbrot Numba")
    s = timer()
    create_fractal_numba(-2.0, 0.5, -1.25, 1.25, image, 100)
    e = timer()
    print(f"{e-s} seconds")
    # plt.show()
    # ##################################################################################
    start_time = time.time()
    mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,200)
    print("Mandelbrot Vectorized Numba --- %s seconds ---" % (time.time() - start_time))
    ####################################################################################
    start_time = time.time()
    arr = mandelbrot_dynamic(xmin, xmax, ymin, ymax, N=600)
    print('\nMandelbrot Multiprocessing --- %s seconds ---' % (time.time() - start_time))






























if __name__ == '__main__':
     main()