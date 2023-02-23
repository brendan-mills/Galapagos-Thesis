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

chnsel="HHZ" # channel code selection

quake = UTCDateTime("2018-06-26T9:15:00.000")
swarm = UTCDateTime("2018-06-26T17:17:00.000")
trem = UTCDateTime("2018-06-26T19:40:00.000")

def plot_time(ax, starttime, time):
    h = time.timestamp - starttime.timestamp
    h = h/3600
    ax.axvline(h, color = 'r')
    return ax


#this line corrects for the quiet day
# t0 = t0-(177-130)*tdur

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

# st = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.Decon.mseed')
t1 = t0 + 16*3600#start of tremor
t2 = t0 + 21*3600#end of tremor
# st.trim(starttime=t1, endtime=t2)

window_duration_sec = 20
average = 20

def calc_covar(st, spec_w = True):
    stream = csn.arraystream.ArrayStream(st)
    raw_stream = stream.copy()
    signal_duration_sec = t2-t1
    stream = raw_stream.copy()
    # synchronize data
    stream = stream.synchronize(start=t1, duration_sec=signal_duration_sec, method="linear")
    # preprocess using smooth spectral whitening and temporal normalization
    stream.preprocess(domain="spectral", method="smooth")
    stream.preprocess(domain="temporal", method="smooth")
    print('Done preprocessing')
    # calculate covariance from stream
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, window_duration_sec, average)
    print('Done with covar matrix')
    # calculate spectral width
    if spec_w:
        spectral_width = covariances.coherence(kind="spectral_width")
        print('Done with spectral width')
        return (times, frequencies, covariances, spectral_width)
    else:
        return (times, frequencies, covariances, None)

# show network covariance matrix spectral width
def plot_covar( spec_w, title = '' ):
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
    plot_time(ax, t1, swarm)
    plot_time(ax,t1, trem)
    plt.savefig('../Figs/'+title+'.jpg', format='jpg', dpi=400, bbox_inches='tight')
    plt.show()
    
files = ['8G.Array..HHZ.2018.177', 
          '8G.Array..HHZ.2018.177.Decon', 
          '8G.Array..HHZ.2018.177.BP1', 
          '8G.Array..HHZ.2018.177.BP2', 
          '8G.Array..HHZ.2018.177.BP3',
          '8G.Array..HHZ.2018.177.BP4']

def crunch_all():
    data = []
    for f in files:
        print('Reading '+f)
        st = opy.read('/Volumes/LaCie/SN_Thesis/Day177/'+f+'.mseed')
        st.trim(starttime=t1, endtime=t2)
        print('Starting covariance calculation')
        spec_w = calc_covar(st)
        data.append(spec_w)
        plot_covar(spec_w, f)
    return data
        
# #average spec widths
# sampling_rate = stream[0].stats.sampling_rate  # assumes all streams have the same sampling rate
# # i_freq_low = round(0.5 * spectral_width.shape[1] / sampling_rate)
# # i_freq_high = round(5 * spectral_width.shape[1] / sampling_rate)
# i_freq_low = 0
# i_freq_high = 2 #check the seydoux papers for more info

# spectral_width_average = np.mean(spectral_width[:, i_freq_low:i_freq_high], axis=1)

# # Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
# covariance_1st = covariances.eigenvectors(covariance=True, rank=0)

# # Extract cross-correlations
# lags, correlation = csn.correlationmatrix.cross_correlation(
#     covariance_1st, sampling_rate
# )
# duration_sec = len(st[0]) / st[0].stats.sampling_rate  # length of stream to be processed, in seconds
# duration_min = duration_sec / 60
# nwin = correlation.nwin()  # number of time windows
# fig,ax = plt.subplots(figsize = (10,5))
# ax.plot(np.linspace(0, duration_min, nwin)/60, spectral_width_average, "r")

end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )
# import os
#os.system('say "Done"')
