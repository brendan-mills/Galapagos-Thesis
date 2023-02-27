#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:10:26 2023

@author: brendanmills
"""
import os
import tqdm
import obspy
from obspy import UTCDateTime
from glob import glob
import shutil

PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_DESTINATION = DIRPATH_RAW #for the mass downloader
os.makedirs(DIRPATH_DESTINATION, exist_ok=True)
t0 = t0 = UTCDateTime("2018-06-26T17:0:00.000")
tdur = 4*3600
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
# Create directory
os.makedirs(DIRPATH_PROCESSED, exist_ok=True)

filepaths_meta = sorted(glob(os.path.join(DIRPATH_RAW, "*.xml")))
for p in filepaths_meta:
    shutil.copy2(p,DIRPATH_PROCESSED)
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed"))) #this gathers a list of the file Paths

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
    print(stream)    
    