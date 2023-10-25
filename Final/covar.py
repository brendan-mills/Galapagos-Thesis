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
import os
from glob import glob
from obspy import UTCDateTime
from tqdm import tqdm
from numpy import linalg as LA

start_timer = time.time()
tdur=3600*24 # duration in seconds for one day
t0 = UTCDateTime("2018-06-26T0:0:00.000")
dec_factor = 4
year = 2018

chnsel="HHZ" # channel code selection
quake = UTCDateTime("2018-06-26T9:15:00.000")
swarm = UTCDateTime("2018-06-26T17:17:00.000")
trem = UTCDateTime("2018-06-26T19:40:00.000")
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/Station_Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
PROCESSED_PATH = PROJECT_PATH + 'Processed/'
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/'
sta_array = ['SN04', 'SN05', 'SN07', 'SN11', 'SN12', 'SN13', 'SN14', 'SN06']

stream = csn.arraystream.read(PROCESSED_PATH + '8GEC.All..HHZ.Decon.filt1.trim.decim.mseed')
stream_c = stream
stats = stream[0].stats
t1 = stats.starttime
t2 = stats.endtime
signal_duration_sec = t2-t1

# st.trim(starttime=t1, endtime=t2)
#%% Side By side Covariance
window_duration_sec = 20
average = 30
stas = [s.stats.station for s in stream.sort(keys=['station'])]
times, frequencies, covariances = csn.covariancematrix.calculate(
    stream, window_duration_sec, average
)
time1 = 24
time2 = 56
freq = 40


# show covariance from first window and first frequency
covariance_show = np.abs(covariances[time1, freq])
fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True, figsize=(13,5))
img1 = ax1.imshow(covariance_show, cmap="Blues")
ax1.set_xticks(np.arange(len(stas)))
ax1.set_xticklabels(stas, rotation=45)
ax1.set_yticks(np.arange(len(stas)))
ax1.set_yticklabels(stas)
ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

covariance_show = np.abs(covariances[time2, freq])
img2 = ax2.imshow(covariance_show, cmap="Reds")
ax2.set_xticks(np.arange(len(stas)))
ax2.set_xticklabels(stas, rotation=45)
ax2.set_yticks(np.arange(len(stas)))
ax2.set_yticklabels(stas)
ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


ax1.set_title((t1+times[time1]).datetime)
ax2.set_title((t1+times[time2]).datetime)

fig.colorbar(img1).set_label("Covariance modulus")
fig.colorbar(img2).set_label("Covariance modulus")
fig.suptitle('Network Covariance Matrix at 1 Hz',fontsize=20)
fig.savefig('/Users/brendanmills/Documents/Senior_Thesis/Figs/Covar/matrixes.svg',format='svg',dpi=300)
#%% Eigenvalues
def lil_sw(eigs):
    sw = 0
    for i,l in enumerate(eigs):
        sw = sw + i*l
    sw = sw / np.sum(eigs)
    return sw

c1 = covariances[time1, freq]
c2 = covariances[time2, freq]

fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True, figsize=(10,4), sharey=True)
w1, v1 = LA.eig(c1)
w1 = np.flip(np.sort(np.abs(w1)))
w1 = w1/w1[0]
ax1.plot(w1,'r.-')

w2, v2 = LA.eig(c2)
w2 = np.flip(np.sort(np.abs(w2)))
w2 = w2/w2[0]
ax2.plot(w2,'r.-')

sw1 = lil_sw(w1)
sw2 = lil_sw(w2)

ax1.axvline(sw1,ls='--')
ax2.axvline(sw2,ls='--')

ax1.text(sw1+0.2,0.8,rf'$\sigma = {round(sw1,2)}$')
ax2.text(sw2+0.2,0.8,rf'$\sigma = {round(sw2,2)}$')

ax1.set_ylabel(r'Normalized Eigenvalue ($\lambda_i / \lambda_1$)')
ax1.set_title('Quiescent - 17:00:00')
ax2.set_title('Active - 18:20:00')


fig.suptitle('Eigenvalue Distribution')
fig.supxlabel('Ordered eigenvalue index')
fig.savefig('/Users/brendanmills/Documents/Senior_Thesis/Figs/Covar/eigs.svg',format='svg',dpi=300)

#%% preprocess using smooth spectral whitening and temporal normalization
window_duration_sec = 60
average = 20
stream = stream.synchronize(start=t1, duration_sec=signal_duration_sec, method="linear")
stream.preprocess(domain="spectral", method="smooth")
stream.preprocess(domain="temporal", method="smooth")

# calculate covariance from stream
times, frequencies, covariances = csn.covariancematrix.calculate(
    stream, window_duration_sec, average
)
# calculate spectral width
spectral_width = covariances.coherence(kind="spectral_width")
print(spectral_width.shape)
#%%
NFFT = 512
[Pxx, freqs, bins, im] = plt.specgram(stream_c[0].data, NFFT=int(NFFT), Fs=stream_c[0].stats.sampling_rate, noverlap=int(NFFT-1))
#%% Display the SW
# show network covariance matrix spectral width
fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(16,14),sharex=True, gridspec_kw={'height_ratios': [1, 2, 3]})

plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

tr = stream_c.select(station='SN07')[0]
stats = tr.stats

x = np.linspace(t1.hour, t2.hour, stats.npts)

ax[0].plot(x,tr,'k')
ax[0].set_ylabel("Z Velocity [m/s]",fontsize=20)
# ax[0].set_title(f'{stats.station}')

img1 = ax[1].pcolormesh(bins/(60*60)+t1.hour, freqs, 10 * np.log10(Pxx), cmap='inferno',shading='auto')
ax[1].set_ylabel("Frequency [Hz]",fontsize=24)
# ax[1].set_title("Spectrogram")

img2 = ax[2].pcolormesh(
    times / 3600 + stats.starttime.hour, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
)

ax[2].set_ylim([0, stream[0].stats.sampling_rate / 2])
ax[2].set_xlabel("Time on 2018-06-26 [hours UTC]",fontsize=24)
ax[2].set_ylabel("Frequency [Hz]",fontsize=24)
# ax[2].set_title("Spectral Width")

plt.colorbar(img2).set_label("Covariance matrix spectral width")
ax[2].set_yscale('log')
ax[2].set_ylim(0.02,10)

fig.savefig(FIG_PATH + 'Covar/sw.png',format='png',dpi=300,transparent=True)
#%%
fig, ax = plt.subplots(figsize=(16,4))
tr = stream_c.select(station='SN07')[0]
stats = tr.stats

x = np.linspace(t1.hour, t2.hour, stats.npts)

ax.plot(x,tr,'k')
ax.set_ylabel("Vertical velocity [m/s]")
ax.set_title(f'{stats.station}')
fig.savefig(FIG_PATH + 'Covar/sw.png',format='png',dpi=300)
def sw_helper(win, avg):
    stream = csn.arraystream.read(PROCESSED_PATH + '8GEC.All..HHZ.Decon.filt1.trim.decim.mseed')
    stream = stream.synchronize(start=t1, duration_sec=signal_duration_sec, method="linear")
    stream.preprocess(domain="spectral", method="smooth")
    stream.preprocess(domain="temporal", method="smooth")
    # calculate covariance from stream
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, win, avg)
    
    # calculate spectral width
    spectral_width = covariances.coherence(kind="spectral_width")
    # show network covariance matrix spectral width
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10,5))
    img = ax.pcolormesh(
        times / 3600 + stats.starttime.hour, frequencies, spectral_width.T,
        rasterized=True, cmap="viridis_r")

    ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
    ax.set_xlabel("2018-06-26 (hours)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectral and temporal preprocessing, {win} sec window, average over {avg} windows")
    plt.colorbar(img).set_label("Covariance matrix spectral width")

    ax.set_ylim(0,4)
    plt.savefig(FIG_PATH + f'SW_Lots/r1w{win}a{avg}')   
    plt.close()
    
def sw_many():
    for w in np.arange(20,31,2):
        for a in tqdm(np.arange(6,31,2), desc = f'window_duration_sec={w}'):
            sw_helper(w,a)

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
    # plot_time(ax, t1, swarm)
    # plot_time(ax,t1, trem)
    plt.savefig('../Figs/'+title+'.jpg', format='jpg', dpi=400, bbox_inches='tight')
    plt.show()

#%% average spec widths
sampling_rate = stream[0].stats.sampling_rate  # assumes all streams have the same sampling rate
low = 0.5
high = 0.75
def sw_avg(low, high):
    i_freq_low = round(low * spectral_width.shape[1] / sampling_rate)
    i_freq_high = round(high * spectral_width.shape[1] / sampling_rate)
    
    spectral_width_average = np.mean(spectral_width[:, i_freq_low:i_freq_high], axis=1)
    return spectral_width_average

def plot_sw_avg(ax, low, high, c_i):
    color = ["orange", 'g', 'b',]
    ax.plot(hours, sw_avg(low, high), color[c_i], label=f'{low} - {high} Hz')
spectral_width_average = sw_avg(low, high)

# Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
# covariance_1st = covariances.eigenvectors(covariance=True, rank=0)

# Extract cross-correlations
# lags, correlation = csn.correlationmatrix.cross_correlation(
#     covariance_1st, sampling_rate
# )
duration_sec = len(stream[0]) / stream[0].stats.sampling_rate  # length of stream to be processed, in seconds
duration_min = duration_sec / 60
# nwin = correlation.nwin()  # number of time windows
nwin = len(spectral_width)
fig,ax = plt.subplots(constrained_layout=True, figsize=(16,8))
hours = np.linspace(0, duration_min, nwin)/60 + stats.starttime.hour
plot_sw_avg(ax, 0.01, 0.02, 0)
plot_sw_avg(ax, 0.5, 1, 1)
plot_sw_avg(ax, 1, 6, 2)
# ax.plot(hours, sw_avg(0.01, 0.02), "orange", label='0.01 - 0.02 Hz')
# ax.plot(hours, sw_avg(0.1, 0.2), "g", label='0.1 - 0.2 Hz')
# ax.plot(hours, sw_avg(0.25, 0.75), "b", label='0.25 - 0.75 Hz')
# ax.axvspan(17.535, 18.58, alpha=0.25, color='red')
# ax.axvspan(19.15, 19.96, alpha=0.25, color='red')

ax.legend()


ax.set_ylabel('Average Spectral Width')
ax.set_xlabel('Hours')
ax.set_title(f'Average spectral width in various bands')
#%% Trace
tr = stream.select(station='SN05')[0]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(tr,'k')
fig.savefig(FIG_PATH + 'trace05.svg',format='svg')
#%% Spectral Width
fig, ax = plt.subplots(figsize=(48,20))

img2 = ax.pcolormesh(
    times / 3600 + stats.starttime.hour, frequencies, spectral_width.T, rasterized=True, cmap="viridis_r"
)

ax.set_ylim([0, stream[0].stats.sampling_rate / 2])
ax.set_xlabel("2018-06-26 (UTC hours)",)
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Spectral Width")

plt.colorbar(img2).set_label("Covariance matrix spectral width")
ax.set_yscale('log')
ax.set_ylim(0.02,10)

fig.savefig(FIG_PATH + 'Covar/spec_w.png',format='png',dpi=300)
#%% Spectrogram

#%% map
