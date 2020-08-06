# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:41:33 2020

@author: JG07DA
"""

import unittest
import numpy as np

import variant as vat

class MandelTest(unittest.TestCase):
    
    def setUp(self):
        # self.methods = ['vat.mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)',
        #                 'vat.mandel_set_numba(xmin,xmax,ymin,ymax,width,height,maxiter)',
        #                 'vat.mandel_set_numba_vect(xmin,xmax,ymin,ymax,width,height,maxiter)',
        #                 'vat.mandel_set_mp(xmin,xmax,ymin,ymax,width,height,maxiter,1)']
        
        self.methods = ['vat.mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)',
                        'vat.mandel_set_numba(xmin,xmax,ymin,ymax,width,height,maxiter)']
        
    def test_mandel_naive_numba(self):
       xmin=-2.0 # Rc lower bound
       xmax=1.0  # Rc upper bound
       ymin=-1.5 # Ic lower bound
       ymax=1.5
       width=500  #Pre
       height=500 #Pim
       maxiter=100 #iterations
       threshold = 2
       
       x = vat.mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)
       
       for m in self.methods:
           y = eval(m)
           np.testing.assert_allclose(x,y)  
           
    def test_mandel_vectorize_mp(self):
        xmin=-2.0 # Rc lower bound
        xmax=1.0  # Rc upper bound
        ymin=-1.5 # Ic lower bound
        ymax=1.5
        width=500  #Pre
        height=500 #Pim
        maxiter=100 #iterations
        threshold = 2
        
        x = vat.mandel_set_naive(xmin,xmax,ymin,ymax,width,height,maxiter, threshold)
       
        for m in self.methods:
            y = eval(m)
            np.testing.assert_allclose(x,y)  
        
           
   
if __name__ == '__main__':
    unittest.main()