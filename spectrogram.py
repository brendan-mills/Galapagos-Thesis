#! /usr/bin/env python

# Matoza example spectrogram plot for COPR data 10/28/2022

import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime

# read data
st = read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP4.mseed')
t0 = UTCDateTime("2018-06-26T0:0:00.000")
t1 = t0 + 17*3600#start of tremor
t2 = t0 + 24*3600#end of tremor
st.trim(starttime=t1, endtime=t2)
#%% Low Freq
# read data
st = read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP4.mseed')
t0 = UTCDateTime("2018-06-26T0:0:00.000")
t1 = t0 + 17*3600#start of tremor
t2 = t0 + 24*3600#end of tremor
st.trim(starttime=t1, endtime=t2)
st.decimate(10) # need to do this in steps otherwise unstable
st.decimate(10)
st.decimate(2)
st.merge(method=1)
st_unfilt = st.copy()
#st.filter("highpass",freq=0.5)


# create time vector
dt = st[0].stats.starttime.timestamp
tBDF = np.linspace(st[0].stats.starttime.timestamp - dt,
                   st[0].stats.endtime.timestamp - dt,
                   st[0].stats.npts)


# figure
fig=plt.figure(1,figsize=(10,6))
#
## plot setup
pthick = 0.4
pgap = 0.02
ptop = 1 - pthick -pgap -0.01
xpos = 0.1
xthick = 0.8
hfont = {'fontname':'Helvetica'}

# Waveform
ax1 = plt.axes([xpos, ptop, xthick, pthick])
ax1.patch.set_facecolor('black')
plt.plot(tBDF/(60*60), st[0].data, color = '#FFFF00', linewidth = 0.3)

plt.xticks([])
plt.yticks(fontsize=14, **hfont)
plt.xlim(0,tBDF[np.size(tBDF)-1]/(60*60))
plt.ylabel('p [Pa]', fontsize=16, **hfont)

# Spectrogram
NFFT = 512
ptop = ptop - pthick -pgap
ax2 = plt.axes([xpos, ptop, xthick, pthick])

# specgram makes a plot so we delete this and replot with pcolormesh
[Pxx, freqs, bins, im] = plt.specgram(st_unfilt[0].data, NFFT=int(NFFT), Fs=st_unfilt[0].stats.sampling_rate, noverlap=int(NFFT-1))

fig.delaxes(ax2)
ax2 = plt.axes([xpos, ptop, xthick, pthick])

plt.pcolormesh(bins/(60*60), freqs, 10 * np.log10(Pxx), cmap='inferno',shading='auto')

plt.ylabel('f [Hz]', fontsize=16)
plt.yticks(fontsize=14, **hfont)
plt.xlabel("Time [hours] since " + str(st[0].stats.starttime), fontsize=16, **hfont)
plt.xlim(0,tBDF[np.size(tBDF)-1]/(60*60))


fname = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Spec/sn_spec_low.jpg'
plt.savefig(fname, format='jpg', dpi=400, bbox_inches='tight')
plt.show()

#%% High Freq
# read data
st = read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP4.mseed')
t0 = UTCDateTime("2018-06-26T0:0:00.000")
t1 = t0 + 17*3600#start of tremor
t2 = t0 + 24*3600#end of tremor
st.trim(starttime=t1, endtime=t2)
st.merge(method=1)
st_unfilt = st.copy()
#st.filter("highpass",freq=0.5)
t0 = UTCDateTime("2018-06-26T0:0:00.000")
t1 = t0 + 17*3600#start of tremor
t2 = t0 + 21*3600#end of tremor
st.trim(starttime=t1, endtime=t2)

# create time vector
dt = st[0].stats.starttime.timestamp
tBDF = np.linspace(st[0].stats.starttime.timestamp - dt,
                   st[0].stats.endtime.timestamp - dt,
                   st[0].stats.npts)


# figure
fig=plt.figure(1,figsize=(10,6))
#
## plot setup
pthick = 0.4
pgap = 0.02
ptop = 1 - pthick -pgap -0.01
xpos = 0.1
xthick = 0.8
hfont = {'fontname':'Helvetica'}

# Waveform
ax1 = plt.axes([xpos, ptop, xthick, pthick])
ax1.patch.set_facecolor('black')
plt.plot(tBDF/(60*60), st[0].data, color = '#FFFF00', linewidth = 0.3)

plt.xticks([])
plt.yticks(fontsize=14, **hfont)
plt.xlim(0,tBDF[np.size(tBDF)-1]/(60*60))
plt.ylabel('v [m/s]', fontsize=16, **hfont)

# Spectrogram
NFFT = 2048
ptop = ptop - pthick -pgap
ax2 = plt.axes([xpos, ptop, xthick, pthick])

# specgram makes a plot so we delete this and replot with pcolormesh
[Pxx, freqs, bins, im] = plt.specgram(st_unfilt[0].data, NFFT=int(NFFT), Fs=st_unfilt[0].stats.sampling_rate, noverlap=int(NFFT/2))

fig.delaxes(ax2)
ax2 = plt.axes([xpos, ptop, xthick, pthick])

plt.pcolormesh(bins/(60*60), freqs, 10 * np.log10(Pxx), cmap='inferno',shading='auto', vmin = -175)

plt.ylabel('f [Hz]', fontsize=16)
plt.yticks(fontsize=14, **hfont)
plt.xlabel("Time [hours] since " + str(st[0].stats.starttime), fontsize=16, **hfont)
plt.xlim(0,tBDF[np.size(tBDF)-1]/(60*60))

# cax2 = plt.axes([0.91, ptop, 0.01, pthick])
# cbar = plt.colorbar()
# cbar.set_label(r'Pa$^2$ /Hz [dB]')
ax2.set_ylim([0,40])

fname = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Spec/sn_spec_high.jpg'
plt.savefig(fname, format='jpg', dpi=400, bbox_inches='tight')
plt.show()

