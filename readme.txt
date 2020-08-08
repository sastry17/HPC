Readme.txt
---------------------------------------------------------------------------------------------------------------------------

Variant.py

This is the main script of the assignment. The different implementations of the Mandelbrot set are executed within the script. The script can be executed with the CLI or in Spyder IDE. 

The execution status is displayed on the console. The plots, simulation data and the speedup plots are generated in the /Output/ directory. 
-----------------------------------------------------------------------------------------------------------------------------
Vectorized.sh

The speedup for the paralled threads execution for the mandelbrot set can be profiled by running the vectorized.sh script. The script can be called with a terminal. The script outputs the execution time for 1,2,4,8 and 15 threads running in parallel. More threads can be added for profiling in the for loop of the script. The script calls the vectorized.py script that contains the implementation for numba-vectorized version that enables parallel execution. 

------------------------------------------------------------------------------------------------------------------------------
Doc.pdf

The doc.pdf contains the documentation specified as per the assignment tasks. The documentation provides an overview of the imlementation, output, algorithm and the profiling information.

----------------------------------------------------------------------------------------------------------------------------
Requirements

The requirements for the scripts execution are:

System requirements:
1. Python 3 (>3.6)
2. PDF reader
3. OS - Windows/OSX/Ubuntu18.04 (Tested on these OS)

Python modules:
1. matplotlib
2. numba
3. numpy
4. multiprocessing
5. os
6. time

-----------------------------------------------------------------------------------------------------------------------------
