# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:59:40 2020

@author: JG07DA
"""
import numpy as np
import matplotlib.pyplot as plt




#set ogrid[Rc, Tc] and 5000 as pre and pim as specified in the project
x,y=np.ogrid[-2:1:5000j,-1.5:1.5:5000j]

print('')
print('Grid set')
print('')

c=x + 1j*y
z=0

for g in range(100):
        #print('Iteration number: ',g)
        z=z**2 + c

threshold = 2
mask=np.abs(z) < threshold

print('')
print('Plotting using imshow()')
plt.imshow(mask.T,extent=[-2,1,-1.5,1.5],cmap=plt.cm.hot)

print('')
print('plotting done')
print('')

plt.gray()

print('')
print('Preparing to render')
print('')

plt.show()