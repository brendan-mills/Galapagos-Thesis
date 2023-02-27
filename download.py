#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:56:37 2023

@author: brendanmills
"""
import cartopy
import os
import tqdm
import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import mass_downloader
from glob import glob
import shutil
from obspy import read_inventory

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
    station="SN04,SN05,SN07,SN11,SN12,SN13,SN14",
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
    station="SN06",
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
        ax.text(station.longitude+0.05, station.latitude, "  " + station.code)
