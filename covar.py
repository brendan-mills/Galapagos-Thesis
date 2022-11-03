#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 01:53:56 2022

@author: brendanmills
"""

import covseisnet as csn
import obspy as opy
import matplotlib.pyplot as plt
import numpy as np
import time
from obspy import UTCDateTime

start_timer = time.time()
tdur=3600*24 # duration in seconds for one day
t0 = UTCDateTime("2018-06-26T0:0:00.000")
dec_factor = 4
year = 2018

def single_mat(stream, window_duration_sec, average):
    channels = [s.stats.channel for s in stream]
    
    # calculate covariance from stream
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, window_duration_sec, average)
    
    # show covariance from first window and first frequency
    covariance_show = np.abs(covariances[0, 0])
    fig, ax = plt.subplots( figsize = (10,5), constrained_layout=True)
    img = ax.imshow(covariance_show, origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_title("Single-station multiple channels covariance")
    plt.colorbar(img, shrink=0.6).set_label("Covariance modulus")

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection

#This code takes files form the drive and only picks out day 177 and then writes them to another folder
list_stations = ['SN04','SN05', 'SN07', 'SN11', 'SN12']
t1 = t0 + tdur

#'/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.onset_tremor.Decon.mseed'
st = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP1.Decon.mseed')
#st.trim(starttime=t0+14*3600)
window_duration_sec = 5
average = 80
stream = csn.arraystream.ArrayStream(st)
raw_stream = stream.copy()

t1 = stream[0].stats.starttime
t2 = stream[0].stats.endtime
signal_duration_sec = t2-t1

# create plot #4 - preprocess data with smooth spectral whitening and temporal normalization
stream = raw_stream.copy()

# downsample data to 25 Hz
stream.decimate(dec_factor)

# synchronize data
stream = stream.synchronize(start=t1, duration_sec=signal_duration_sec, method="linear")

# preprocess using smooth spectral whitening and temporal normalization
stream.preprocess(domain="spectral", method="smooth")
stream.preprocess(domain="temporal", method="smooth")
print('Done preprocessing')
# calculate covariance from stream
times, frequencies, covariances = csn.covariancematrix.calculate(
    stream, window_duration_sec, average
)
print('Done with covar matrix')
# calculate spectral width
spectral_width = covariances.coherence(kind="spectral_width")
print('Done with spectral width')
# show network covariance matrix spectral width
fig, ax = plt.subplots(figsize = (10,5), constrained_layout=True)
img = ax.pcolormesh(
    times / 3600, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
)
# ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
ax.set_ylim([0.5, 4])
ax.set_xlabel("Hours")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(f"Spectral width with preprocessing, Win-{window_duration_sec}sec, avg-{average}")
plt.colorbar(img).set_label("Covariance matrix spectral width")

#average spec widths
sampling_rate = stream[0].stats.sampling_rate  # assumes all streams have the same sampling rate
i_freq_low = round(0.5 * spectral_width.shape[1] / sampling_rate)
i_freq_high = round(5 * spectral_width.shape[1] / sampling_rate)

spectral_width_average = np.mean(spectral_width[:, i_freq_low:i_freq_high], axis=1)

# Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
covariance_1st = covariances.eigenvectors(covariance=True, rank=0)

# Extract cross-correlations
lags, correlation = csn.correlationmatrix.cross_correlation(
    covariance_1st, sampling_rate
)
duration_sec = len(st[0]) / st[0].stats.sampling_rate  # length of stream to be processed, in seconds
duration_min = duration_sec / 60
nwin = correlation.nwin()  # number of time windows
fig,ax = plt.subplots(figsize = (10,5))
ax.plot(np.linspace(0, duration_min, nwin)/60, spectral_width_average, "r")


end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )
#print("\a")
import os
os.system('say "Done"')
