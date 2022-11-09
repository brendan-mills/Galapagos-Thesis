#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:22:46 2022

@author: brendanmills
"""
import numpy as np
import matplotlib.pyplot as plt
import obspy as opy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import time
import os
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger

start_timer = time.time()
t0 = UTCDateTime("2018-06-26T0:0:00.000")
tdur=3600*24 # duration in seconds for one day

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN11" # station code selection (* for all)
year = 2018
info = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)
client = Client("IRIS")

st_bp = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP1.Decon.mseed')
st_d = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.Decon.mseed')
st_raw = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.mseed')

st = st_bp

t1 = t0 + 9*3600#start of tremor
t2 = t0 + 9.5*3600#end of tremor
st.trim(starttime=t1, endtime=t2)
# st[2].plot()

trace = st[2]
df = trace.stats.sampling_rate

cft = classic_sta_lta(trace.data, int(10 * df), int(100 * df))
plot_trigger(trace, cft, 5, 0.5)

