#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 04:23:23 2023

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
from tqdm.notebook import tqdm





def main():
    PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
    DIRPATH_RAW = PROJECT_PATH + 'Raw/'
    DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
    TTIMES_PATH = PROJECT_PATH + 'TTimes/Sparse'
    DEM_PATH = '/Users/brendanmills/Documents/Senior_Thesis/GalapagosDEM/'
    FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
    t0 = t0 = UTCDateTime("2018-06-26T17:0:00.000")
    tdur = 4*3600

    domain = mass_downloader.RectangularDomain(
        minlatitude=-1,#south
        maxlatitude=-0.5794,#north
        minlongitude=-91.3508,#west
        maxlongitude=-90.9066,#east
    )
    # grid extent Longitude: 55.67° to 55.81° (145 points), Latitude: -21.3° to -21.2° (110 points)
    lon_min = domain.minlongitude
    lon_max = domain.maxlongitude
    lat_min = domain.minlatitude
    lat_max = domain.maxlatitude

    depth_min = -1.5
    depth_max = 20


    all_sta = ['SN04', 'SN05','SN07','SN11','SN12','SN13','SN14','SN06']
    sta = ['SN04','SN07','SN13','SN12','SN14','SN06']
    items3 = [(0.5, 10, 120, 30, 0.5, sta, 'pRun14'),
             (0.5, 4, 120, 30, 0.5, sta, 'pRun15'),
             (4, 8, 120, 30, 0.5, sta, 'pRun16'),
             (0.5, 2, 120, 30, 0.5, sta, 'pRun17'),
             (2, 4, 120, 30, 0.5, sta, 'pRun18'),
             (6, 8, 120, 30, 0.5, sta, 'pRun19'),
             (8, 10, 120, 30, 0.5, sta, 'pRun20'),
        ]
    
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(corr_and_plot, items3)

if __name__ == "__main__":
    main()
    
    
    
    
#%%
'''
items = [(0.5, 10, 120, 30, 0.5, all_sta, 'pRun0'),
         (0.5, 4, 120, 30, 0.5, all_sta, 'pRun1'),
         (4, 8, 120, 30, 0.5, all_sta, 'pRun2'),
         (0.5, 2, 120, 30, 0.5, all_sta, 'pRun3'),
         (2, 4, 120, 30, 0.5, all_sta, 'pRun4'),
         (6, 8, 120, 30, 0.5, all_sta, 'pRun5'),
         (8, 10, 120, 30, 0.5, all_sta, 'pRun6'),
    ]

sta = ['SN04', 'SN05','SN07','SN11','SN12','SN14','SN06']
items2 = [(0.5, 10, 120, 30, 0.5, sta, 'pRun7'),
         (0.5, 4, 120, 30, 0.5, sta, 'pRun8'),
         (4, 8, 120, 30, 0.5, sta, 'pRun9'),
         (0.5, 2, 120, 30, 0.5, sta, 'pRun10'),
         (2, 4, 120, 30, 0.5, sta, 'pRun11'),
         (6, 8, 120, 30, 0.5, sta, 'pRun12'),
         (8, 10, 120, 30, 0.5, sta, 'pRun13'),
    ]

sta = ['SN04','SN07','SN13','SN12','SN14','SN06']
items3 = [(0.5, 10, 120, 30, 0.5, sta, 'pRun7'),
         (0.5, 4, 120, 30, 0.5, sta, 'pRun8'),
         (4, 8, 120, 30, 0.5, sta, 'pRun9'),
         (0.5, 2, 120, 30, 0.5, sta, 'pRun10'),
         (2, 4, 120, 30, 0.5, sta, 'pRun11'),
         (6, 8, 120, 30, 0.5, sta, 'pRun12'),
         (8, 10, 120, 30, 0.5, sta, 'pRun13'),
    ]
'''