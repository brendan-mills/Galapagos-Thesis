#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:56:37 2023

@author: ?
"""
import os
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from pykonal.solver import PointSourceSolver
from pykonal.transformations import geo2sph
import covseisnet as csn
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import read, read_inventory
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
## TO DO
MAX_DEPTH = 5
VERT_DENSITY = 20
LAT_DENSITY = 300

PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
TTIMES_PATH = PROJECT_PATH + f'TTimes/{LAT_DENSITY}lat{VERT_DENSITY}dep/'
os.makedirs(TTIMES_PATH, exist_ok=True)
domain = mass_downloader.RectangularDomain(
    minlatitude=-1.17,#south
    maxlatitude=-0.45,#north
    minlongitude=-91.45,#west
    maxlongitude=-90.8,#east
)

lon_min = domain.minlongitude
lon_max = domain.maxlongitude
lat_min = domain.minlatitude
lat_max = domain.maxlatitude
t0 = t0 = UTCDateTime("2018-06-26T17:0:00.000")
tdur = 4*3600

DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'

with open(TTIMES_PATH + 'info.txt', 'w') as f:
    f.write(f'vert_density {VERT_DENSITY}\n')
    f.write(f'vert_density {LAT_DENSITY}\n')
    f.write(f'lonmin {lon_min}')
    f.write(f'lonax {lon_max}')
    f.write(f'latmin {lat_min}')
    f.write(f'latmax {lat_max}')
#%% Velovity Model

FILEPATH_VELOCITY = '/Users/brendanmills/Documents/Senior_Thesis/Data/1dvmod.txt'
# Read velocity model
velocity_layers = pd.read_csv(
    FILEPATH_VELOCITY, 
    names=["depth", "P", "S"],
    index_col="depth",
    delimiter= ' ',
    )
velocity_layers = velocity_layers[-3:MAX_DEPTH]
# Show table
#print(velocity_layers)

ax = velocity_layers.plot(
    drawstyle="steps-post",
    ylabel="Wave Velocity (km/s)",
    xlabel="Depth (km)",
    title="1D velocity model from Karabulut et al. 2011",
    grid=True,
    xlim=(-3,MAX_DEPTH)
)
plt.scatter(velocity_layers.index, velocity_layers.loc[:,'P'])
#%% Interpolate Depths
#OK I know this is code soup

interp_depths = np.linspace(np.amin(velocity_layers.index), np.amax(velocity_layers.index), VERT_DENSITY)
#this smashes the arrays together
interp_depths = np.sort(np.concatenate((interp_depths,np.array(velocity_layers.index)),axis=None))
#This removes duplicates at the ends
interp_depths = interp_depths[1:-1]

velocity_layers_interp = velocity_layers.reindex(interp_depths)
velocity_layers_interp = velocity_layers_interp.interpolate()

depth_min = -1.5  # edifice summit (km)
depth_max = MAX_DEPTH  # max depth of traveltime grids (km)
depths = np.linspace(depth_max, depth_min, VERT_DENSITY)
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

longitudes = np.linspace(domain.minlongitude, domain.maxlongitude, LAT_DENSITY)
# sample latitudes in decreasing order to get corresponding colatitudes in increasing order (see explanation further)
latitudes = np.linspace(domain.minlatitude, domain.maxlatitude, LAT_DENSITY)
# Xarray
velocities = velocity_layers_interp.stack().to_xarray()
velocities = velocities.rename({"level_1": "phase"})

# Add longitudes and latitude dimensions (automatically broadcast vector into 3D array)
velocities = velocities.expand_dims(latitude=latitudes, longitude=longitudes)
# Stations
# Get inventories
inventory = read_inventory(DIRPATH_PROCESSED + '*xml')

# Extract stations
stations = [sta for net in inventory for sta in net]
attrs = "longitude", "latitude", "elevation", "code"
stations = [{item: getattr(sta, item) for item in attrs} for sta in stations]

stations = list(np.append(stations,[{'longitude': -91.409849, 'latitude': -0.791115, 'elevation': 46.796349, 'code': 'CEAZ'},
                  {'longitude': -91.01927, 'latitude': -0.8597631, 'elevation': 267.847, 'code': 'PVIL'},
                  {'longitude': -91.1134240, 'latitude': -0.7824234, 'elevation': 1067.3101983, 'code': 'VCH1'},
                  {'longitude': -90.9701114, 'latitude': -0.4548725, 'elevation': 134.2029101, 'code': 'ALCE'},
                  ]))

# Turn into dataframe
network = pd.DataFrame(stations).set_index("code")
network["depth"] = -1e-3 * network.elevation

# Select slice at first latitude index
velocities_slice = velocities.sel(phase="P", latitude=latitudes.min())

# Show velocities
img = velocities_slice.T.plot.imshow(cmap="RdBu", add_colorbar=False, figsize=(15, 12))#change
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
#%% Travel Times
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
        

os.makedirs(TTIMES_PATH, exist_ok=True)
travel_times.to_netcdf(TTIMES_PATH + 'travel_times.nc')
for s in list(network.index):
    tt = travel_times.sel(station=s,phase='P').T#I added the Transpose here, not sure if it is right
    nptt = tt.to_numpy()
    nptt = np.flip(nptt, axis=2)
    np.save(TTIMES_PATH  + f'{s}.npy',nptt)
#%%
CONTOUR_LEVELS = 20
SEISMIC_PHASE = "P"
station = network.loc["ALCE"]

# Show
latitude_id = np.abs(travel_times.latitude - station.latitude).argmin()
time_delays = travel_times.sel(phase=SEISMIC_PHASE, station=station.name)
time_delays = np.flip(time_delays, 1)
time_delays = time_delays.isel(latitude=latitude_id)
img = time_delays.plot.contourf(
    add_colorbar=False, cmap="RdPu", levels=CONTOUR_LEVELS, figsize=(20, 2)
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


