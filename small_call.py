#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:16:29 2023

@author: brendanmills
"""
import cartopy
import os
import numpy as np
import tqdm
import covseisnet as csn
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import obspy
from obspy.clients.fdsn import Client
from obspy import read, read_inventory, UTCDateTime
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
from osgeo import gdal
import time
from calc_plot import *
import multiprocessing as mp

plt.ioff()#this should supress output
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
tdur = 4*3600
low = 0.5
high = 10
win = 20
av = 12
ov = 0.5
sta_array = ['SN04', 'SN05', 'SN07', 'SN11', 'SN12', 'SN13', 'SN14', 'SN06']
norm = 'frame'
name = '20s12a'
test = False

# corr_and_plot(low, high, win, av, ov, all_sta, name, norm, test=True)
os.makedirs(FIG_PATH + name, exist_ok=False)
print(low, high, win, av, ov, sta_array, norm, name)
st, inv = get_streams(sta_array)
print(st)
stream_out = stream_pre_process(st, low, high, win, av, ov)
correlation = calc_correl(stream_out, win, av)
nwin=correlation.nwin()
with open(FIG_PATH + name +'/' + 'info.txt', 'w') as f:
    f.write(f'Nwin {nwin}\n')
    f.write(f'Run {name}\n')
    f.write(f'Low {low}\n')
    f.write(f'High {high}\n')
    f.write(f'Window_sec {win}\n')
    f.write(f'Avgerage {av}\n')
    f.write(f'Overlap {ov}\n')
    f.write(f'Stations {str(sta_array)}\n')
    f.write(f'Norm {norm}\n')
#%%
beam, df = calc_beam(stream_out, correlation,low, high, sigma=20, test=test)
df.to_csv(FIG_PATH + name + '/max.csv')
#%%

for i in tqdm.tqdm(range(nwin)):
    fieldxy, xz, yz = get_field(beam, i, norm)
    plot_map(fieldxy, tdur/nwin,stream_out, inv, i, name)
    try:
        plot_depth(xz, yz, i, name, beam)
    except:
        print('I screwed up the Depth')
    if test:
        break
print()
print(name)