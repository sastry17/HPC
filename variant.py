# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:28:54 2020

@author: JG07DA
"""

import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import colors
import os
import time
import multiprocessing as mp
from numba import jit, vectorize, complex64, int32


#######################################################################################################
#Naive

def linspace(start, stop, n):
    step = float(stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def mandel_naive(c, maxiter, threshold):
    z = c
    for n in range(maxiter):
       if abs(z) > 2:
            return n
       z = z*z + c
    return n

def mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold):
    r = linspace(xmin, xmax, width)
    i = linspace(ymin, ymax, height)
    n = [[0]*width for _ in range(height)]
    for x in range(width):
        for y in range(height):
            n[y][x] = mandel_naive(complex(r[x], i[y]), maxiter, threshold)
    
    return n

#########################################################################################################
#Numba

@jit(nopython=True)
def mandel_numba(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return n

def mandel_set_numba(xmin,xmax,ymin,ymax,width,height,maxiter):
    r = linspace(xmin, xmax, width)
    i = linspace(ymin, ymax, height)
    n = [[0]*width for _ in range(height)]
    for x in range(width):
        for y in range(height):
            n[y][x] = mandel_numba(complex(r[x], i[y]), maxiter)
    
    
    return n


#########################################################################################################
#Numba - Algo Improved

@jit(nopython=True)
def mandel_numba_adv(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0: #squared modulus
            return n
        imag = 2 * real*imag + cimag
        real = real2 - imag2 + creal       
    return n

@jit(nopython=True)
def mandel_adv_algo(xmin,xmax,ymin,ymax,width,height,maxiter):
    r = np.linspace(xmin, xmax, width)
    i = np.linspace(ymin, ymax, height)
    n = np.empty((height, width), dtype=np.int32)
    for x in range(width):
        for y in range(height):
            n[y, x] = mandel_numba_adv(r[x], i[y], maxiter)
    return n
#########################################################################################################
#Numba Vectorize

@vectorize([int32(complex64, int32)], target='parallel')
def mandel_numba_vect(c, maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0: #squared modulus
            return n
    return n
        
def mandel_set_numba_vect(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n = mandel_numba_vect(c,maxiter)
    return n
#########################################################################################################
#Multiprocessing
#ncpus = 8 #set number of processing units here
@jit(nopython=True)
def mandel_mp(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0: #squared modulus
            return n
        imag = 2 * real*imag + cimag
        real = real2 - imag2 + creal       
    return n

@jit(nopython=True)
def mandel_mp_row(args):
    y, xmin, xmax, width, maxiter = args
    r = np.linspace(xmin, xmax, width)
    res = [0] * width
    for x in range(width):
        res[x] = mandel_mp(r[x], y, maxiter)
    return res
        

def mandel_set_mp(xmin,xmax,ymin,ymax,width,height,maxiter,ncpus):
    i = np.linspace(ymin, ymax, height)
    with mp.Pool(ncpus) as pool:
        n = pool.map(mandel_mp_row, ((a, xmin, xmax, width, maxiter) for a in i))
    return n

#########################################################################################################
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
#########################################################################################################
def results(name,arr,xmin,xmax,ymin,ymax):
    plt.imshow(arr, extent=[xmin, xmax, ymin, ymax], cmap='hot')
    plt.suptitle(name)
    #path = "./Output/temp/"
    #plt.savefig(path+name+".pdf")
    plt.show()
    #np.savetxt(path+name+'.csv', arr, delimiter=',') 
#########################################################################################################
def table(naive_run,numba_vect_run,mp_run,m):
    fig = plt.figure(dpi=80)
    ax = fig.add_subplot(1,1,1)
    table_data=[
    ["Naive-Python", naive_run],
    ["Numba Vectorized", numba_vect_run],
    ["Multiprocessing ("+str(m)+" CPUs)" , mp_run],
    ]
    table = ax.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')
    path = "./Output/temp/"
    plt.savefig(path+"ExecutionTime.pdf")    
    plt.show()

#########################################################################################################

def main():
    xmin=-2.0 # Rc lower bound
    xmax=1.0  # Rc upper bound
    ymin=-1.5 # Tc lower bound
    ymax=1.5  # Tc upper bound
    width=1000  #Pre
    height=1000 #Pim
    maxiter=100 #iterations
    threshold = 4 #threshold T
    m = mp.cpu_count()
    
    thisrun = time.strftime("%Y%m%d-%H%M%S")
    createFolder('./Output/temp')
       
    
    start_time = time.time()
    plot_naive=mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)
    naive_run= float(time.time() - start_time)
    #print('\nMandelbrot Naive--- %s seconds ---' % (time.time() - start_time))
    
    # # start_time = time.time()
    # # mandel_set_numba(xmin,xmax,ymin,ymax,width,height,maxiter)
    # # print('\nMandelbrot Numba--- %s seconds ---' % (time.time() - start_time))
    
    # # start_time = time.time()
    # #plot_numba=mandel_adv_algo(xmin,xmax,ymin,ymax,width,height,maxiter)
    # # print('\nMandelbrot Numba Algo Improved--- %s seconds ---' % (time.time() - start_time))
    
    start_time = time.time()
    plot_numba_vect=mandel_set_numba_vect(xmin,xmax,ymin,ymax,width,height,maxiter)
    numba_vect_run= (time.time() - start_time)
    #print('\nMandelbrot Numba Vectorized--- %s seconds ---' % (time.time() - start_time))
    
    start_time = time.time()
    plt_mp=mandel_set_mp(xmin,xmax,ymin,ymax,width,height,maxiter,1)
    mp_run = (time.time() - start_time)
    #print('\nMandelbrot Multiprocessing --- %s seconds ---' % (time.time() - start_time))
    mp_name="Multiprocessing "+str(m)+"(CPUs)"
    
    results("Naive",plot_naive,xmin,xmax,ymin,ymax)
    results("Numba_Vectorized",plot_numba_vect,xmin,xmax,ymin,ymax)
    results(mp_name,plt_mp,xmin,xmax,ymin,ymax)
    table(naive_run,numba_vect_run,mp_run,1)
    os.rename("./Output/temp","./Output/"+thisrun)
   # mandelbrot_image(mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold))
    ####################################################################################################
    # run for multiple processors as per the machine
    
    for m in range(1, mp.cpu_count()+1):
         start_time = time.time()
         plt_mp=mandel_set_mp(xmin,xmax,ymin,ymax,width,height,maxiter,m)
         mp_run = (time.time() - start_time)
         mp_name="Multiprocessing "+str(m)+"(CPUs)"
         print(mp_name+" "+str(mp_run)+" seconds")
         
    ######################################################################################################     
         
def mandelbrot_image(fun, xmin=-2.0, xmax=0.5, ymin=-1.25, ymax=1.25, width=1000, height=1000, maxiter=80, cmap='magma'):
        dpi = float(width) / 7.0
        z = fun(xmin, xmax, ymin, ymax, width, height, maxiter)
    
        fig, ax = plt.subplots(figsize=(7,7), dpi=72)
        plt.xticks([])
        plt.yticks([])
        plt.title("[{xmin}, {ymin}] to [{xmax}, {ymax}]".format(**locals()))
    
        norm = colors.PowerNorm(0.3)
        ax.imshow(z,cmap=cmap,origin='lower',norm=norm)
        
    

    
    
    
    
#########################################################################################################

if __name__ == '__main__':
     main()
#########################################################################################################