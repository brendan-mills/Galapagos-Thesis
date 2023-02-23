#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:56:37 2023

@author: brendanmills
"""
import cartopy
import os
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from pykonal.solver import PointSourceSolver
from pykonal.transformations import geo2sph
import covseisnet as csn
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import read, read_inventory
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
import elevation
from osgeo import gdal
## TO DO
#get the other stations from EC network


#%% DOWNLOADER
#the first step is to download the data
#first set up some path names
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_DESTINATION = DIRPATH_RAW #for the mass downloader
os.makedirs(DIRPATH_DESTINATION, exist_ok=True)

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
t0 = t0 = UTCDateTime("2018-06-26T17:0:00.000")
tdur = 4*3600
restrictions = mass_downloader.Restrictions(
    starttime=t0,
    endtime=t0 + tdur,
    #chunklength_in_sec=60*60,
    network="8G",
    location="*",
    channel="*HZ",
    station="SN01,SN03,SN04,SN05,SN07,SN08,SN11,SN12,SN13,SN14",
    reject_channels_with_gaps=True,
    minimum_length=0.0,
    minimum_interstation_distance_in_m=1.0,
    channel_priorities=["HH[ZNE]", "BH[ZNE]"],
    )

restrictions2 = mass_downloader.Restrictions(
    starttime=t0,
    endtime=t0 + tdur,
    #chunklength_in_sec=60*60,
    network="EC",
    location="*",
    channel="*HZ",
    station="SN02,SN06,SN09,SN10,ALCE",
    reject_channels_with_gaps=True,
    minimum_length=0.0,
    minimum_interstation_distance_in_m=1.0,
    channel_priorities=["HH[ZNE]", "BH[ZNE]"],
    )

# Downloader instance
downloader = mass_downloader.MassDownloader(providers=['IRIS'])

# Download
downloader.download(
    domain,
    restrictions,
    mseed_storage=DIRPATH_DESTINATION,
    stationxml_storage=DIRPATH_DESTINATION,
)
downloader.download(
    domain,
    restrictions2,
    mseed_storage=DIRPATH_DESTINATION,
    stationxml_storage=DIRPATH_DESTINATION,
)

#%% STATIONS DOMAIN
MAP_EXTENT = (
    domain.minlongitude,
    domain.maxlongitude,
    domain.minlatitude,
    domain.maxlatitude,
)

# Read inventory
inventory = read_inventory(os.path.join(DIRPATH_DESTINATION, "*.xml"))

# Create axes
ax = plt.axes(projection=cartopy.crs.PlateCarree())
ax.set_extent(MAP_EXTENT)
ax.gridlines(draw_labels=True)
ax.coastlines()

# Show
for network in inventory:
    for station in network:
        ax.plot(station.longitude, station.latitude, "kv")
        #ax.text(station.longitude, station.latitude, "  " + station.code)
#%% Set More paths and pre-preprocess
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
# Create directory
os.makedirs(DIRPATH_PROCESSED, exist_ok=True)

# Copy meta to destination
#!cp {DIRPATH_RAW}*xml {DIRPATH_PROCESSED} #this line came from ipython
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed"))) #this gathers a list of the file Paths

stream = obspy.read(filepaths_raw[0])
stream.trim(endtime=t0+tdur)
print(stream)
stream.plot(size=(600, 250), show=True)
plt.gcf().set_facecolor('w')
#%% PREPROCESSING
TARGET_SAMPLING_RATE = 25.0
MAX_SEGMENTS = 10
FREQ_MIN = 2.0
FREQ_MAX = 12.0
TAPER_PERCENT = 0.02
TAPER_TYPE = "cosine"
MSEED_ENCODING = "FLOAT64"
TARGET_STARTTIME = t0
TARGET_ENDTIME = t0+tdur

for filepath_waveform in tqdm.tqdm(filepaths_raw, desc="Processing data"):
    # Read trace
    trace = obspy.read(filepath_waveform)[0]

    # Split trace into segments to process them individually
    stream = trace.split()

    # Apply detrend on segments
    stream.detrend("constant")
    stream.detrend("linear")
    stream.taper(TAPER_PERCENT, type=TAPER_TYPE)

    # Merge traces, filling gaps with zeros and imposing start and end times
    stream = stream.merge(fill_value=0.0)
    trace = stream[0]
    trace.trim(starttime=TARGET_STARTTIME, endtime=TARGET_ENDTIME, pad=True, fill_value=0.0)

    # Resample at target sampling rate
    trace.decimate(4)

    # Attach instrument response
    filepath_inventory = f"{trace.stats.network}.{trace.stats.station}.xml"
    filepath_inventory = os.path.join(DIRPATH_RAW, filepath_inventory)
    inventory = obspy.read_inventory(filepath_inventory)
    trace.attach_response(inventory)

    # Remove instrument gain
    trace.remove_sensitivity()

    # Detrend
    trace.detrend("constant")
    trace.detrend("linear")
    trace.taper(TAPER_PERCENT, type=TAPER_TYPE)

    # Filter
    trace.filter(
        "bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, zerophase=True
    )
    trace.taper(TAPER_PERCENT, type=TAPER_TYPE)

    # Write processed traces
    _, filename = os.path.split(filepath_waveform)
    filepath_processed_waveform = os.path.join(DIRPATH_PROCESSED, filename)
    trace.write(filepath_processed_waveform, encoding=MSEED_ENCODING)
#%% Compare
# Get a file to read
filepath_raw = filepaths_raw[0]
filename_raw = os.path.basename(filepath_raw)
filepath_processed = os.path.join(DIRPATH_PROCESSED, filename_raw)

# Loop over cases and show
for filepath in (filepath_raw, filepath_processed):
    stream = obspy.read(filepath)
    stream.trim(endtime=t0+tdur)
    stream.plot(size=(600, 250))    
    
#%% Velovity Model
FILEPATH_VELOCITY = '/Users/brendanmills/Documents/Senior_Thesis/Data/1dvmod.txt'
# Read velocity model
velocity_layers = pd.read_csv(
    FILEPATH_VELOCITY, 
    names=["depth", "P", "S"],
    index_col="depth",
    delimiter= ' ',
    )
velocity_layers = velocity_layers[-3:20]
# Show table
print(velocity_layers)

ax = velocity_layers.plot(
    drawstyle="steps-post",
    ylabel="Wave Velocity (km/s)",
    xlabel="Depth (km)",
    title="1D velocity model from Karabulut et al. 2011",
    grid=True,
    xlim=(-3,20)
)
plt.scatter(velocity_layers.index, velocity_layers.loc[:,'P'])
#%% Velocity Model 2
#OK I know this is code soup
interp_depths = np.linspace(np.amin(velocity_layers.index), np.amax(velocity_layers.index), 100)
#this smashes the arrays together
interp_depths = np.sort(np.concatenate((interp_depths,np.array(velocity_layers.index)),axis=None))
#This removes duplicates at the ends
interp_depths = interp_depths[1:-1]

velocity_layers_interp = velocity_layers.reindex(interp_depths)
velocity_layers_interp = velocity_layers_interp.interpolate()

depth_min = -1.5  # edifice summit (km)
depth_max = 20  # max depth of traveltime grids (km)
depths = np.linspace(depth_max, depth_min, 60)
velocity_layers_interp = velocity_layers_interp.reindex(depths,method='ffill')

velocity_layers.plot()
velocity_layers_interp.plot(
    drawstyle="steps-post",
    xlabel="Depth (km)",
    ylabel="Speed (km/s)",
    title="1D velocity model from Karabulut et al. 2011",
    ax=plt.gca(),
    grid=True,
    figsize=(15, 12),
    marker="s",
    ls=""
)

# Labels and legends
plt.axvspan(depths.min(), depths.max(), alpha=0.2)
plt.legend(["P", "S", "P interpolated", "S interpolated", "Domain"])
plt.show()
#%% Expand Model Laterally
longitudes = np.linspace(domain.minlongitude, domain.maxlongitude, 150)
# sample latitudes in decreasing order to get corresponding colatitudes in increasing order (see explanation further)
latitudes = np.linspace(domain.minlatitude, domain.maxlatitude, 150)
# Xarray
velocities = velocity_layers_interp.stack().to_xarray()
velocities = velocities.rename({"level_1": "phase"})

# Add longitudes and latitude dimensions (automatically broadcast vector into 3D array)
velocities = velocities.expand_dims(latitude=latitudes, longitude=longitudes)
