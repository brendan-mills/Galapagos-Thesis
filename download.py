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
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/Final/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_DESTINATION = PROJECT_PATH + 'Processed/' #for the mass downloader

t0 = UTCDateTime("2018-06-26T12:0:00.000")
tdur = 24*3600


# Read inventory
inventory = read_inventory(os.path.join(DIRPATH_DESTINATION, "*.xml"))


