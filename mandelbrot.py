# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:40:15 2020

@author: JG07DA
"""

from timeit import default_timer as timer
import numpy as np
import time
from numba import jit,  guvectorize,  complex64, int32
import matplotlib.pyplot as plt
import math
import multiprocessing as mp

def mandelbrot_mp(c, max_iterations=100, threshold=2):
    z = 0
    for i in range(max_iterations):
        z = z**2 + c
        if z.real**2 + z.imag**2 > threshold:
            return i+1
        if i == max_iterations-1:
            return 0

def mandelbrot_naive(z,iterations, threshold):
    c = z
    for n in range(iterations):
        if abs(z) > threshold:
            return n
        z = z*z + c
    return iterations

def mandelbrot_set_naive(width,height,min_x,max_x,min_y,max_y,maxIterations):
    r1 = np.linspace(min_x, max_x, width)
    r2 = np.linspace(min_y, max_y, height)
    return ((r1,r2,[mandelbrot_naive(complex(r, i),maxIterations,2) for r in r1 for i in r2]))


@jit
def mandelbrot_numba(x, y, max_iters):

    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 2:
            return i/max_iters

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
        if real * real + imag * imag > 2.0:
            return n/maxiter
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


ncpus=4

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

    with mp.Pool(ncpus) as pool:
        result = pool.map(mandelbrot_mp, myList, 2)

    return np.reshape(np.array(result), (N,N))



def results(name,arr,xmin,xmax,ymin,ymax):
    plt.imshow(arr, extent=[xmin, xmax, ymin, ymax], interpolation="bilinear", cmap=plt.cm.hot)
    plt.suptitle(name)
    path = "./Output/temp/"
    plt.savefig(path+name+".pdf")
    plt.show()
    np.savetxt(path+name+'.csv', arr, delimiter=',') 




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
    naive=mandelbrot_set_naive(1000,1000,-2.0,0.5,-1.25,1.25,100)
    print("Mandelbrot Naive --- %s seconds ---" % (time.time() - start_time))
   # results("Naive",naive,xmin,xmax,ymin,ymax)
    ######################################################################################
    #print("Mandelbrot Numba")
    start_time = time.time()
    numba=create_fractal_numba(-2.0, 0.5, -1.25, 1.25, image, 100)
    e = timer()
    print("Mandelbrot Numba: ----- %s second s-----"% (time.time() - start_time))
    
    ####################################################################################
    start_time = time.time()
    vectorized=mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,200)
    print("Mandelbrot Vectorized Numba --- %s seconds ---" % (time.time() - start_time))
    ####################################################################################
    start_time = time.time()
    multiplrocessing = mandelbrot_dynamic(xmin, xmax, ymin, ymax, N=600)
    print('Mandelbrot Multiprocessing --- %s seconds, ---' % (time.time() - start_time))
    





























if __name__ == '__main__':
     main()