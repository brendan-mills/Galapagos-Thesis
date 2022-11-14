#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:35:50 2022

@author: brendanmills
"""
import numpy as np
import matplotlib.pyplot as plt

dat = np.load('/Users/brendanmills/Documents/Senior_Thesis/RobinExamp/BON.npy')

sta = (1,1,1)

x = np.linspace(0., 1., 50)
y = np.linspace(0., 1., 50)
z = np.linspace(0., 1., 50)

# xs, ys = np.meshgrid(x, y, sparse=True)
# zs = np.sqrt(xs**2 + ys**2)
# print(xs.shape, ys.shape, zs.shape)

# h = plt.contourf(x, y, zs)
# plt.axis('scaled')
# plt.colorbar()
# plt.show()

xs, ys, zs = np.meshgrid(x, y, z, indexing='ij', sparse = True)
ts = np.sqrt(2*xs**2 + ys**2 + zs**2)
print(xs.shape, ys.shape, zs.shape, ts.shape)

h = plt.contourf(x, y, ts[1], levels = np.linspace(0,2,100))
plt.axis('scaled')
plt.colorbar()
plt.show()



