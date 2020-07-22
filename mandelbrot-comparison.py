# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:38:29 2020

@author: JG07DA
"""


import time, math, numpy as np, matplotlib.pyplot as plt, multiprocessing as mp 
from numba import jit


def mandelbrot(c, max_iterations=100):
    z = 0
    for i in range(max_iterations):
        z = z**2 + c
        if z.real**2 + z.imag**2 > 4:
            return i+1
        if i == max_iterations-1:
            return 0


def mandelbrot_naive(xmin, xmax, ymin, ymax, N=100):
    incx= math.fabs((xmax-xmin)/N) #returns abs value for x
    incy= math.fabs((ymax-ymin)/N) #returns abs value for y
    x, y, myList, n, arr = xmin, ymax, [], 1, np.ndarray(shape=(0, N))
    while y > ymin and n <= N:
        while x < xmax:
            i=mandelbrot(complex(x,y), 100)
            myList.append(i)
            x += incx
        arr = np.append(arr, np.array([myList[:N]]), axis=0)
        x = xmin
        y -= incy
        myList=[]
        n += 1
    return arr

@jit
def mandelbrot_numba(c, max_iterations=100):
    z = 0
    for i in range(max_iterations):
        z = z**2 + c
        if z.real**2 + z.imag**2 > 4:
            return i+1
        if i == max_iterations-1:
            return 0




@jit
def mandelbrot_vec_numba(xmin, xmax, ymin, ymax, N=100):
    incx= math.fabs((xmax-xmin)/N) #returns abs value for x
    incy= math.fabs((ymax-ymin)/N) #returns abs value for y
    x, y, myList, n, arr = xmin, ymax, np.array([]), 1, np.zeros(shape=(0, N))
    while y > ymin and n <= N:
        while x < xmax:
            i=mandelbrot_numba(complex(x,y), 100)
            myList.append(i)
            x += incx
        arr = np.append(arr, np.array([myList[:N]]), axis=0)
        x = xmin
        y -= incy
        myList=[]
        n += 1
    return arr




def call_mandelbrot_naive(xmin, xmax, ymin, ymax, N, results, i):
    results.put({i:mandelbrot_naive(xmin, xmax, ymin, ymax, N)})
 


def mandelbrot_static(xmin, xmax, ymin, ymax, N=100):
    results = mp.Queue()
    processes = []
    count = mp.cpu_count() // 2
    arr_upper = np.ndarray(shape=(0, N//count))
    arr_lower = np.ndarray(shape=(0, N//count))
    incx = math.fabs((xmax - xmin) / count)

    x, i = xmin, 0
    while i < mp.cpu_count():
        i += 1
        if i == count+1:
            x = xmin   # refresh x to xmin

        if i > count:
            p = mp.Process(target=call_mandelbrot_naive, args=(x, x + incx, ymin, 0, N//count, results, i))
            p.start()
            processes.append(p)
            x += incx
        else:
            p = mp.Process(target=call_mandelbrot_naive, args=(x, x + incx, 0, ymax, N//count, results, i))
            p.start()
            processes.append(p)
            x += incx


    result_set = {}
    for p in processes:
        result_set.update(results.get())

    i = 1

    for p in processes:
        if i <= count:
            if i == 1:
                arr_upper = np.vstack([arr_upper, result_set[i]])
            else:
                arr_upper = np.concatenate((arr_upper, result_set[i]), axis=1)
        else:
            if i == count + 1:
                arr_lower = np.vstack([arr_lower, result_set[i]])
            else:
                arr_lower = np.concatenate((arr_lower, result_set[i]), axis=1)
        i+=1

    for p in processes:
        p.join()

    return np.append(arr_upper, arr_lower, axis=0)


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
        result = pool.map(mandelbrot, myList)

    return np.reshape(np.array(result), (N,N))


def main():
    xmin= -2.0
    xmax= 1.0
    ymin= -1.5
    ymax= 1.5

    # print('mandelbrot_naive, Resolution 600x600:')
    # start = time.time()
    # arr  = mandelbrot_naive( xmin, xmax, ymin, ymax, N=600)
    # end = time.time()
    # print(f'{end - start} secs')
    
    
    print('mandelbrot_vec_numba, Resolution 600x600:')
    start = time.time()
    arr  = mandelbrot_vec_numba( xmin, xmax, ymin, ymax, N=600)
    end = time.time()
    print(f'{end - start} secs')

    #  print('\nmandelbrot parallel static, Resolution 600x600:')
    #  start = time.time()
    #  arr = mandelbrot_static(xmin, xmax, ymin, ymax, N=600)
    #  end = time.time()
    #  print(f'{end - start} secs')


    # print('\nmandelbrot parallel dynamic, Resolution 600x600:')
    # start = time.time()
    # arr = mandelbrot_dynamic(xmin, xmax, ymin, ymax, N=600)
    # end = time.time()
    # print(f'{end - start} secs')
    #plt.imshow(arr, extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.hot) 
    # plt.imshow(np.log(arr), extent=[xmin, xmax, ymin, ymax], cmap=plt.cm.hot)
    # plt.savefig('MandelBrot.png')
    # plt.show()


if __name__ == '__main__':
     main()