#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:03:37 2023

@author: brendanmills
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from obspy import UTCDateTime, read_inventory, read
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import covseisnet as csn
import obspy
from glob import glob
import tqdm
from osgeo import gdal
import rasterio
import scipy
import pygmt

ttmodel = '100lat20dep'
TRACK_PATH = f'/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/{ttmodel}/max.csv'
DEM_PATH = '/Users/brendanmills/Documents/Senior_Thesis/GalapagosDEM/'
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/Station_Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
PROCESSED_PATH = PROJECT_PATH + 'Processed/'
META_PATH = PROJECT_PATH + 'Response/'
t0 = UTCDateTime("2018-06-26T17:0:00.000")

filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed")))
filepaths_meta = sorted(glob(os.path.join(META_PATH,'*')))

# download metadata
inv = obspy.Inventory()
net = {"lat": [], "lon": []}
for p in filepaths_meta:
    inv_to_be = read_inventory(p)
    inv.extend(inv_to_be)
    net["lat"].append(inv_to_be[0][0].latitude)
    net["lon"].append(inv_to_be[0][0].longitude)
print(pd.DataFrame(net))


# grid extent Longitude: 55.67° to 55.81° (145 points), Latitude: -21.3° to -21.2° (110 points)
# lon_min = -91.3508
# lon_max = -90.9066
# lat_min = -1
# lat_max = -0.5794
stream = csn.arraystream.read(PROCESSED_PATH + '8GEC.All..HHZ.Decon.filt1.trim.decim.mseed')
stats = stream[0].stats

# Define region of interest around Sierra Negra

region = [-91.45, -90.8, -1.17, -0.45]
# region = [-91.25, -91, -0.9, -0.7]

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
    frame=["SWrt+tTremor Movement", "xa0.1", "ya0.1"],
    cmap=True,
    transparency=25,
)

track = pd.read_csv(TRACK_PATH)
#plot the path
pygmt.makecpt(cmap="viridis", series=[track.index.min(), track.index.max()])
fig.plot(x=track['long'],
          y=track['lat'],
          style="cc",
          size = 0.15*np.ones(len(track)),
          fill=track.index,
          pen="black",
          cmap=True
          )

fig.plot(x=net['lon'],
          y=net['lat'],
          style="t0.3t",
          # size = 0.15*np.ones(len(track)),
          fill='white',
          pen="black",
          )

with fig.inset(
    position="jTR+o0.1c",
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

fig.colorbar(frame="af+lTime Window")
fig.show()


