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

start_timer = time.time()

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN11" # station code selection (* for all)
tdur=3600*24 # duration in seconds for one day
#t0 = UTCDateTime("2018-06-26T0:0:00.000")
dec_factor = 4
year = 2018
info2 = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)

def get_stream(start_day, end_day, net, sta, chn, year):
    #the code here uses the decon_name file names but they raw from the raw data!
    print('Getting stream')
    if type(sta) == str:
        sta = [sta]
    if type(chn) == str:
        chn = [chn]
    info2 = '{}.{}..{}.{}'.format(net, sta[0], chn[0], year)
    days = np.arange(start_day, end_day+1)
    decon_name = '/Volumes/BigGirl/SIERRA_NEGRA/Deconvolved/{}.{}.Decon.mseed'.format(info2,start_day)
    #decon_name = '/Volumes/BigGirl/SIERRA_NEGRA/SN_GALAPAGOS{}.{}.mseed'.format(info2,start_day)
    st  = opy.read(decon_name)
    for jday in days:
        for s in sta:
            for c in chn:
                info2 = '{}.{}..{}.{}'.format(net, s, c, year)
                #decon_name = '/Volumes/BigGirl/SIERRA_NEGRA/SN_GALAPAGOS/{}.{}.mseed'.format(info2,jday)
                decon_name = '/Volumes/BigGirl/SIERRA_NEGRA/Deconvolved/{}.{}.Decon.mseed'.format(info2,jday)
                try:
                    st2 = opy.read(decon_name)
                    st = st + st2
                except:
                    print('I don\'t have the stream for {} jday {}'.format(info2, jday))
    print('Done loading streams')
    st.merge(fill_value = 'interpolate')
    print('Done with merge')
    return st

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

def cov_spectral(stream, window_duration_sec, average):
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, window_duration_sec, average)
    print('Done with covariance matrix')
    # calculate spectral width
    spectral_width = covariances.coherence(kind="spectral_width")
    print('Done computing spectral width')
    # show covariance at first time window and first frequency
    fig, ax = plt.subplots(figsize = (10,5), constrained_layout=True)
    img = ax.pcolormesh(
        times/3600, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
    )
    ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
    ax.set_xlabel("Times (hours)")
    ax.set_ylabel("Frequency (Hz)")
    start_jday = stream[0].stats.starttime.julday
    end_jday = stream[0].stats.endtime.julday-1
    ax.set_title('Days {} to {}, win = {}, avg = {}'.format(start_jday, end_jday, window_duration_sec,average))
    plt.colorbar(img).set_label("Covariance matrix spectral width")

def cov_spec_pre(stream, window_duration_sec, average):
    stream = csn.arraystream.ArrayStream(stream)
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
    print('Done dinewith spectral width')
    # show network covariance matrix spectral width
    fig, ax = plt.subplots(figsize = (10,5), constrained_layout=True)
    img = ax.pcolormesh(
        times / 3600, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
    )
    ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
    ax.set_xlabel("Hours")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Smooth spectral and temporal preprocessing")
    plt.colorbar(img).set_label("Covariance matrix spectral width")


#single_mat(stream, 10.0,5)
stream = get_stream(177,178, '8G', ['SN11', 'SN07', 'SN04', 'SN12'], ['HHZ'], 2018)
#stream = get_stream(176,178, '8G', ['SN11'], ['HHZ', 'HHN','HHE'], 2018)
#stream.decimate(10)
#cov_spectral(stream, 240,15)
cov_spec_pre(stream, 10,5)




end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )
#print("\a")
import os
os.system('say "ding dong"')
