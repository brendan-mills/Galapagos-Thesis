#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:20:45 2023

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

depth_min = -1.5
depth_max = 5
sampling_rate = 25.0

win_duration_sec = 30
win_avg = 8
ovlp=0.5

low_pass = 1/win_duration_sec
high_pass = 12

ttmodel = '300lat20dep'
name = ttmodel + f'_w{win_duration_sec}a{win_avg}'
print(name)
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Station_Data/Raw/'
PROCESSED_PATH = PROJECT_PATH + 'Station_Data/Processed/'
TTIMES_PATH = PROJECT_PATH + 'TTimes/' + ttmodel
META_PATH = PROJECT_PATH + 'Station_Data/Response/'
FIG_PATH = f'/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/{name}/'
BEAM_PATH = PROJECT_PATH + 'Beam/'
COVAR_PATH = PROJECT_PATH + f'Covariances/{name}/'
t0 = UTCDateTime("2018-06-26T17:0:00.000")

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

#%% load in streams and preprocess
stream = csn.arraystream.read(PROCESSED_PATH + '8GEC.All..HHZ.Decon.filt1.trim.decim.mseed')
starttime = stream[0].stats.starttime
endtime = stream[0].stats.endtime
tdur = endtime - starttime

print('Preprocess')
stream.merge(method=1, fill_value="interpolate", interpolation_samples=-1)

## synchronize traces in the stream
stream = stream.synchronize(t0, tdur, method="linear")

## filtering
stream.detrend(type="demean")
stream.detrend(type="linear")
stream.filter(type="bandpass", freqmin=low_pass, freqmax=high_pass)

preproc_spectral_secs = win_duration_sec * win_avg * ovlp
stream.preprocess(
    domain="spectral", method="onebit", window_duration_sec=preproc_spectral_secs
)
stream.trim(starttime, endtime) 

#%% Calculate coherence
print('Covariance, Correlation and Beam')
os.makedirs(COVAR_PATH, exist_ok=True)
try:
    times = np.load(COVAR_PATH + 'times.npy')
    frequencies = np.load(COVAR_PATH + 'frequencies.npy')
    covariances = np.load(COVAR_PATH + 'covariances.npy').view(csn.CovarianceMatrix)
    print('Loaded Covariance')
except:
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, win_duration_sec, win_avg
    )
    np.save(COVAR_PATH + 'times.npy', times)
    np.save(COVAR_PATH + 'frequencies.npy', frequencies)
    np.save(COVAR_PATH + 'covariances.npy', covariances)
print('First Eigen')
# Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
try:
    covariance_1st = csn.covariancematrix.CovarianceMatrix(np.load(COVAR_PATH + 'cov1st.npy'))
    print('Loaded 1st Eigenvectors')
except:
    covariance_1st = covariances.eigenvectors(covariance=True, rank=0)
    np.save(COVAR_PATH + 'cov1st.npy', covariance_1st)
print('Calculating Correlation')
# Extract cross-correlations
lags, correlation = csn.correlationmatrix.cross_correlation(
    covariance_1st, sampling_rate
)
print('Beam from', TTIMES_PATH)
ttimes = csn.traveltime.TravelTime(stream, TTIMES_PATH)
# Initiate beam object and set geographical extent of grid
nwin = correlation.nwin()  # number of time windows
beam = csn.beam.Beam(nwin, ttimes)
beam.set_extent(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
# beam.set_extent(-91.3, -91, -0.9, -0.7, depth_min, depth_max)

#%% Location
try:
    raise
    df = pd.read_csv(FIG_PATH + 'max.csv')
    print('Loaded max.csv')
except:
    df = pd.DataFrame(columns=['long','lat','depth'])
    
    for i in tqdm.tqdm(range(nwin)):
        correl = correlation[i]
        # Filter correlation
        correl = correl.bandpass(low_pass, high_pass, sampling_rate)
    
        # Smooth correlation
        correl = correl.hilbert_envelope()
        correl = correl.smooth(sigma=20)  # default sigma is 5
    
        beam.calculate_likelihood(correl, sampling_rate, i)
    
        beam_max = beam.max_likelihood(i)
        df.loc[i] = [beam_max[0], beam_max[1], beam_max[2]]
    
    
    os.makedirs(FIG_PATH, exist_ok=True)
    print('Write to: ' + FIG_PATH + 'max.csv')
    df.to_csv(FIG_PATH + 'max.csv')
    print('Write to: ' + PROJECT_PATH + 'beam_temp.npy')
    np.save(PROJECT_PATH + 'beam_temp.npy', beam)
    print('Done')
#%% Field plotting
i_win = 3
region = [-91.45, -90.8, -1.17, -0.45]#whole domain
# Choose the last window for plotting likelihood
likelihood_xyz = beam.likelihood[i_win, :, :, :]

# Take slices at point of max likelihood
i_max, j_max, k_max = np.unravel_index(likelihood_xyz.argmax(), likelihood_xyz.shape)
likelihood_xy = likelihood_xyz[:, :, k_max]

# Normalize likelihood between 0 and 1
field = (likelihood_xy - likelihood_xy.min()) / (
    likelihood_xy.max() - likelihood_xy.min()
)
fig = pygmt.Figure()
fig.basemap(region=region, projection="X10c", frame=True)

#  the position 1/1
# on a basemap, scaled up to be 3 cm wide and draw a rectangular border
# around the image
# pygmt.makecpt(cmap="gray", series=[0, 1])
# fig.grdimage(
#     grid=xarray.DataArray(likelihood_xy.T),
#     projection="M12c",
#     cmap=True,
# )
# fig.coast(
#           water="azure1", 
#           region=region,
#           projection='M12c',
#           frame=["SWrt+tStation Map", "xa0.2", "ya0.2"],
#           )

# fig.show()
#%% Plotting Maximums
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed")))
filepaths_meta = sorted(glob(os.path.join(META_PATH,'*')))

inventory = read_inventory(META_PATH + '*xml')
# Extract stations
stations = [sta for net in inventory for sta in net]
attrs = "longitude", "latitude", "elevation", "code"
stations = [{item: getattr(sta, item) for item in attrs} for sta in stations]

stations = list(np.append(stations,[{'longitude': -91.409849, 'latitude': -0.791115, 'elevation': 46.796349, 'code': 'CEAZ'},
                  {'longitude': -91.01927, 'latitude': -0.8597631, 'elevation': 267.847, 'code': 'PVIL'},
                  {'longitude': -91.1134240, 'latitude': -0.7824234, 'elevation': 1067.3101983, 'code': 'VCH1'},
                  {'longitude': -90.9701114, 'latitude': -0.4548725, 'elevation': 134.2029101, 'code': 'ALCE'},
                  ]))

net = pd.DataFrame(stations)

stats = stream[0].stats

# Define region of interest around Sierra Negra
region = [-91.45, -90.8, -1.17, -0.45]#whole domain
region = [-91.25, -91, -0.9, -0.7]#zoomed in on caldera
# region = [-91.2, -91.09, -0.87, -0.75]#more zoom

# Load sample grid (3 arc-seconds global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
grid = pygmt.grdclip(grid, below=[1, -2000])
# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

# --------------- plotting the hillshade map -----------

pygmt.makecpt(cmap="gray", series=[-1.5, 0.3, 0.01])
fig.grdimage(
    region=region,
    grid=dgrid,
    projection="M12c",
    frame=["SWrt+tTremor Location", "xa0.1", "ya0.1"],
    cmap=True,
    transparency=25,
)

track = pd.read_csv(FIG_PATH + 'max.csv')
#plot the path
dot_time = track.index/nwin*(endtime.hour-starttime.hour) + starttime.hour
pygmt.makecpt(cmap="viridis", series=[dot_time.min(), dot_time.max()])
fig.plot(x=track['long'],
          y=track['lat'],
          style="cc",
          size = 0.15*np.ones(len(track)),
          fill=dot_time,
          pen="black",
          cmap=True
          )
fig.colorbar(frame="af+lTime [hours UTC]")
fig.plot(x=net['longitude'],
          y=net['latitude'],
          style="t0.3c",
          # size = 0.15*np.ones(len(track)),
          fill='white',
          pen="black",
          )
#this plots te fissures
fissure_paths = sorted(glob(os.path.join(PROJECT_PATH + 'Fissures/', "*.txt")))
for file in fissure_paths:
    with open(file) as f:
        lines = np.array([line.strip().split() for line in f.readlines()])
        x = lines[:,0]
        y = lines[:,1]
        fig.plot(
            x=x,
            y=y,
            pen="2p,red",
        )
with open(PROJECT_PATH + 'Fissures/fissure_text.txt') as f:
    lines = np.array([line.strip().split() for line in f.readlines()])
    name = lines[:,0]
    x = lines[:,1]
    y = lines[:,2]
    fig.text(text=name, x=x, y=y)    

with fig.inset(
    position="jTR+o0.2c",
    box="+gwhite+p1p",
    region = [-92, -89, -1.5, 0.7],
    projection="M12/3c",
):
    # Highlight the Japan area in "lightbrown"
    # and draw its outline with a pen of "0.2p".
    fig.coast(land='lightgray',
              water='azure1'
    )
    # Plot a rectangle ("r") in the inset map to show the area of the main
    # figure. "+s" means that the first two columns are the longitude and
    # latitude of the bottom left corner of the rectangle, and the last two
    # columns the longitude and latitude of the uppper right corner.
    rectangle = [[region[0], region[2], region[1], region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="1.5p,blue")



fig.show()
fig.savefig(FIG_PATH + '/map.pdf')
#%% Array Figure
fig = pygmt.Figure()
region = [-91.8, -90.6, -1.07, 0.25]#whole domain
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
# grid = pygmt.grdclip(grid, below=[1, -2000])
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
grid = pygmt.grdclip(grid, below=[1, -2000])
# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

pygmt.makecpt(cmap="gray", series=[-1.5, 0.3, 0.01])
fig.grdimage(
    region=region,
    grid=dgrid,
    projection="M12c",
    cmap=True,
    transparency=25,
)
fig.coast(
          water="azure1", 
          region=region,
          projection='M12c',
          frame=["SWrt+tStation Map", "xa0.2", "ya0.2"],
          )
stations = [sta for net in inventory for sta in net]
sta_temp = [{item: getattr(sta, item) for item in attrs} for sta in stations]

sta_perm = [{'longitude': -91.409849, 'latitude': -0.791115, 'elevation': 46.796349, 'code': 'CEAZ'},
                  {'longitude': -91.01927, 'latitude': -0.8597631, 'elevation': 267.847, 'code': 'PVIL'},
                  {'longitude': -91.1134240, 'latitude': -0.7824234, 'elevation': 1067.3101983, 'code': 'VCH1'},
                  {'longitude': -90.9701114, 'latitude': -0.4548725, 'elevation': 134.2029101, 'code': 'ALCE'},
                  ]
net_temp = pd.DataFrame(sta_temp)
net_perm = pd.DataFrame(sta_perm)
fig.plot(x=net_temp['longitude'],
          y=net_temp['latitude'],
          style='t0.3c',
          fill='green',
          pen="black",
          label='Temporary Stations'
          )
fig.plot(x=net_perm['longitude'],
          y=net_perm['latitude'],
          style='i0.3c',
          fill='red',
          pen="black",
          label='Permanent Stations'
          )

with fig.inset(position="jTR+w3c+o0.1c", margin=0):
    # Create a figure in the inset using coast. This example uses the azimuthal
    # orthogonal projection centered at 47E, 20S. The land color is set to
    # "gray" and Madagascar is highlighted in "red3".
    fig.coast(
        region="g",
        projection="G-91/-1/?",
        land="gray",
        water="white",
        dcw="MG+gred3",
    )
    fig.plot(
        x=-91, y=-45, style='v0.6c+e', direction=([90], [1]), pen="2p", fill="black"
    )

# fig.colorbar(frame=["a4000f2000", "x+lElevation", "y+lm"])
fig.legend(position="jTL+w3c+o0.1c")
fig.show() 
fig.savefig(PROJECT_PATH + '/stations.pdf')  

