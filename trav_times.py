#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:35:50 2022

@author: brendanmills
"""
import numpy as np
import matplotlib.pyplot as plt
import obspy.taup as taup

depth = np.arange(1000)
pvel = np.linspace(2637,7308,1000, dtype = 'int')
svel = np.linspace(1522,4219,1000, dtype = 'int')
dens = np.linspace(2217,2861,1000, dtype = 'int')

file = open('vel_1D.tvel','w')
for i in range(len(depth)):
    file.write(str(depth[i]) + ' ' + str(pvel[i]) + ' ' + str(svel[i]) + ' ' + str(dens[i]) + '\n')
file.close()



#%%
velmodel_tvel = '/Users/brendanmills/Documents/Senior_Thesis/Code/vel_1D.tvel'
vel_path = 'taup_model'
# taup.taup_create.build_taup_model(velmodel_tvel, output_folder=vel_path)
taup.taup_create.build_taup_model(velmodel_tvel)
velmodel_npz = './velocity_file.npz'
velmodel = taup.TauPyModel(model=velmodel_npz)

#%%
dat = np.load('/Users/brendanmills/Documents/Senior_Thesis/RobinExamp/BON.npy')

sta = (2,5,7)

x = np.arange(0,10,0.1)
y = np.arange(0,10,0.1)
z = np.arange(0,10,0.1)

xs, ys, zs = np.meshgrid(x, y, z, indexing='ij', sparse = True)
ts = np.sqrt( (xs - sta[0])**2 + (ys - sta[1])**2 + (zs - sta[2])**2)
print(xs.shape, ys.shape, zs.shape, ts.shape)

for az  in z:
    h = plt.contourf(x, y, ts[:,:,int(az*len(z))], levels = np.linspace(0,16,17))
    plt.axis('scaled')
    plt.title('z =  ' + str(az))
    plt.colorbar()
    plt.show()
    




