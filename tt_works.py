#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:03:21 2023

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
import shutil

PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data2/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_DESTINATION = DIRPATH_RAW #for the mass downloader
os.makedirs(DIRPATH_DESTINATION, exist_ok=True)
domain = mass_downloader.RectangularDomain(
    minlatitude=-1,#south
    maxlatitude=-0.5794,#north
    minlongitude=-91.3508,#west
    maxlongitude=-90.9066,#east
)
#%%

FILEPATH_VELOCITY = '/Users/brendanmills/Documents/Senior_Thesis/PdFTestRun/jgrb50189-sup-0003-vmodel1d'
# Read velocity model
velocity_layers = pd.read_csv(
    FILEPATH_VELOCITY, 
    names=["depth", "P", "S"],
    index_col="depth",
    )

# Show table
print(velocity_layers)

ax = velocity_layers.plot(
    drawstyle="steps-post",
    ylabel="Wave Velocity (km/s)",
    xlabel="Depth (km)",
    title="1D velocity model from Karabulut et al. 2011",
    grid=True,
    xlim=(-5,5)
)
plt.scatter(velocity_layers.index, velocity_layers.loc[:,'P'])

#OK I know this is code soup
interp_depths = np.linspace(np.amin(velocity_layers.index), np.amax(velocity_layers.index), 100)
#this smashes the arrays together
interp_depths = np.sort(np.concatenate((interp_depths,np.array(velocity_layers.index)),axis=None))
#This removes duplicates at the ends
interp_depths = interp_depths[1:-1]

velocity_layers_interp = velocity_layers.reindex(interp_depths)
velocity_layers_interp = velocity_layers_interp.interpolate()

depth_min = -1.5  # edifice summit (km)
depth_max = 7  # max depth of traveltime grids (km)
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

longitudes = np.linspace(domain.minlongitude, domain.maxlongitude, 150)
# sample latitudes in decreasing order to get corresponding colatitudes in increasing order (see explanation further)
latitudes = np.linspace(domain.minlatitude, domain.maxlatitude, 150)
# Xarray
velocities = velocity_layers_interp.stack().to_xarray()
velocities = velocities.rename({"level_1": "phase"})

# Add longitudes and latitude dimensions (automatically broadcast vector into 3D array)
velocities = velocities.expand_dims(latitude=latitudes, longitude=longitudes)

DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
# Create directory
os.makedirs(DIRPATH_PROCESSED, exist_ok=True)
# Get inventories
inventory = read_inventory(DIRPATH_PROCESSED + '*xml')

# Extract stations
stations = [sta for net in inventory for sta in net]
attrs = "longitude", "latitude", "elevation", "code"
stations = [{item: getattr(sta, item) for item in attrs} for sta in stations]

# Turn into dataframe
network = pd.DataFrame(stations).set_index("code")
network["depth"] = -1e-3 * network.elevation 

## Select slice at first latitude index
velocities_slice = velocities.sel(phase="P", latitude=latitudes.min())

# Show velocities
img = velocities_slice.T.plot.imshow(cmap="RdBu", add_colorbar=False, figsize=(15, 12))
cb = plt.colorbar(img)

# Show stations
plt.plot(network.longitude, network.depth, "wv")

# Labels
ax = plt.gca()
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Depth (km)")
ax.set_title("Velocity model slice from 3D grid")
cb.set_label(f"{velocities_slice.phase.data} velocity (km/s)")
ax.invert_yaxis()
#%%
STATION_ENTRIES = ["latitude", "longitude", "depth"]

# Initialize travel times
velocities_solver = velocities.transpose("phase", "depth", "latitude", "longitude")
travel_times = velocities_solver.copy()
travel_times = travel_times.expand_dims(station=network.index.values).copy()

# Reference point
reference_point = geo2sph((latitudes.max(), longitudes.min(), depths.max()))
node_intervals = (
    np.abs(depths[1] - depths[0]),
    np.deg2rad(np.abs(latitudes[1] - latitudes[0])),
    np.deg2rad(longitudes[1] - longitudes[0]),
)

# Loop over stations and phases
for phase in travel_times.phase.data:
    for station in tqdm.tqdm(network.index, desc=f"Travel times {phase}"):

        # Initialize Eikonal solver
        solver = PointSourceSolver(coord_sys="spherical")
        solver.velocity.min_coords = reference_point
        solver.velocity.node_intervals = node_intervals
        velocity = velocities_solver.sel(phase=phase).to_numpy()
        solver.velocity.npts = velocity.shape
        solver.velocity.values = velocity.copy()
        
        # Source
        src_loc = network.loc[station][STATION_ENTRIES].values
        solver.src_loc = np.array(geo2sph(src_loc).squeeze())

        # Solve Eikonal equation
        solver.solve()

        # Assign to dataarray
        locator = dict(station=station, phase=phase)
        tt = solver.tt.values
        tt[np.isinf(tt)] = 0
        travel_times.loc[locator] = tt
        
TTIMES_PATH = PROJECT_PATH + 'TTimes/'
os.makedirs(TTIMES_PATH, exist_ok=True)
travel_times.to_netcdf(TTIMES_PATH + 'travel_times.nc')

CONTOUR_LEVELS = 20
SEISMIC_PHASE = "P"
station = network.loc["SN13"]

# Show
latitude_id = np.abs(travel_times.latitude - station.latitude).argmin()
time_delays = travel_times.sel(phase=SEISMIC_PHASE, station=station.name)
time_delays = np.flip(time_delays, 1)
time_delays = time_delays.isel(latitude=latitude_id)
img = time_delays.plot.contourf(
    add_colorbar=False, cmap="RdPu", levels=CONTOUR_LEVELS, figsize=(15, 5)
)

# Colorbar
cb = plt.colorbar(img)
cb.set_label(f"Travel times {SEISMIC_PHASE} (seconds)")

# Station
plt.plot(station.longitude, station.depth, "k.")

# Labels
ax = plt.gca()
ax.invert_yaxis()
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Depth (km)")
ax.set_title(f"Travel times from the seismic station {station.name}")
plt.show()