#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:15:08 2023

@author: brendanmills
"""

import os
import pygmt
import numpy as np
import tqdm
import covseisnet as csn
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import obspy
from obspy import read_inventory, UTCDateTime
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
from osgeo import gdal
import pandas as pd
import xarray
import beepy

PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/LongData/'
DIRPATH_DECON = PROJECT_PATH + 'Decon/'
DIRPATH_COVAR = PROJECT_PATH + 'Covars/'

t0 = UTCDateTime("2018-06-26T00:0:00.000")
tday = 24*3600

domain = mass_downloader.RectangularDomain(
    minlatitude=-1.17,#south
    maxlatitude=-0.45,#north
    minlongitude=-91.45,#west
    maxlongitude=-90.8,#east
)
# grid extent Longitude: 55.67° to 55.81° (145 points), Latitude: -21.3° to -21.2° (110 points)
lon_min = domain.minlongitude
lon_max = domain.maxlongitude
lat_min = domain.minlatitude
lat_max = domain.maxlatitude
#%% Functions
def plot_covar( spec_w, title ,t1):
    (times, frequencies, covariances, spectral_width) = spec_w
    if spectral_width is None:
        print('There is no spectral width to plot')
        return
    fig, ax = plt.subplots(figsize = (10,5), constrained_layout=True)
    img = ax.pcolormesh(
        times / 3600, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
    )
    # ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
    
    ax.set_xlabel(f"Time [hours] since {t1}")
    ax.set_ylabel("Frequency (Hz)")
    title = f"Spectral width with preprocessing, Win-{window_duration_sec}sec, avg-{average} {title}"
    ax.set_title(title)
    # ax.set_yscale('log')
    ax.set_ylim([0, 4])
    plt.colorbar(img).set_label("Covariance matrix spectral width")
    # plot_time(ax, t1, swarm)
    # plot_time(ax,t1, trem)
    plt.savefig('../../Figs/'+title+'.jpg', format='jpg', dpi=400, bbox_inches='tight')
    plt.show()
#%% Read in
filepaths_decon = sorted(glob(os.path.join(DIRPATH_DECON, "F1.D.*.mseed")))

stream = obspy.core.Stream()
for f in tqdm.tqdm(filepaths_decon):
    st_in = obspy.read(f)
    stream.append(st_in[0])

#%% Preprocess for covariance plots
stream = csn.arraystream.ArrayStream(stream)

print('Merge')
stream.merge(method=1, fill_value="interpolate", interpolation_samples=-1)
stream_c = stream.copy()
stats = stream[0].stats
t1 = stats.starttime
t2 = stats.endtime

#%% calculate covar for a week
window_duration_sec = 60
average = 20

for day_index in range(31):
    stream = stream_c.copy()
    print(day_index)
    starttime = t0 + day_index*tday
    endtime = starttime + tday
    try:
        npzfile = np.load(DIRPATH_COVAR + f'covarPkg.{day_index}.{window_duration_sec}.{average}.npz')
        times = npzfile['arr_0']
        frequencies = npzfile['arr_1']
        covariances = npzfile['arr_2']
        spectral_width = npzfile['arr_3']
        print('Loaded covars from file')
    except:
        print('Preprocess')
        ## synchronize traces in the stream and preprocesss
        stream = stream.synchronize(start=starttime, duration_sec=tday, method="linear")
        stream.preprocess(domain="spectral", method="smooth")
        stream.preprocess(domain="temporal", method="smooth")
        ##some cleanup
        stream.detrend(type="demean")
        stream.detrend(type="linear")
        stream.trim(starttime, endtime)
        
        print('Calculating covariances')
        times, frequencies, covariances = csn.covariancematrix.calculate(
            stream, window_duration_sec, average
        )
        # calculate spectral width
        spectral_width = covariances.coherence(kind="spectral_width")
        
        np.savez(DIRPATH_COVAR + f'covarPkg.{day_index}.{window_duration_sec}.{average}.npz', times,frequencies, covariances, spectral_width)
    
    plot_covar( (times, frequencies, covariances, spectral_width),'test', starttime)


#%% Plot them together

title = 'stitch_test'
times = np.array([])
spectral_width = np.empty( (0,2999) )
for day_index in range(18):

    npzfile = np.load(DIRPATH_COVAR + f'covarPkg.{day_index}.{window_duration_sec}.{average}.npz')
    times = np.append(times, npzfile['arr_0']+day_index*tday)
    spectral_width = np.append(spectral_width, npzfile['arr_3'], axis=0)

fig, ax = plt.subplots(figsize = (10,5), constrained_layout=True)
img = ax.pcolormesh(
    times[0:spectral_width.shape[0]+1] / 3600/24, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
)
ax.set_xlabel(f"Time [days] since {t1}")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(title)
# ax.set_yscale('log')
ax.set_ylim([0, 4])
plt.colorbar(img).set_label("Covariance matrix spectral width")
# plot_time(ax, t1, swarm)
# plot_time(ax,t1, trem)
plt.savefig('../../Figs/'+title+'.jpg', format='jpg', dpi=400, bbox_inches='tight')
plt.show()