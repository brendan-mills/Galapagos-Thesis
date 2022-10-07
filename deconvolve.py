#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:05:49 2022

@author: brendanmills
"""
import numpy as np
import matplotlib.pyplot as plt
import obspy as opy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import time

start_timer = time.time()
t0 = UTCDateTime("2018-06-26T0:0:00.000")
tdur=3600*24 # duration in seconds for one day

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN11" # station code selection (* for all)
year = 2018
info = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)
client = Client("IRIS")

day = 1

def create_padded_stream(start_day, end_day, padding_hrs=3, info=info):
    #this creates a stream for deconvolving what has padding on either side
    start_day_j = t0.julday + start_day
    end_day_j = t0.julday + end_day
    
    st = opy.read(f'/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{info}.{start_day_j-1}.mseed')
    for d in range(start_day_j, end_day_j):
        st = st + opy.read(f'/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{info}.{d}.mseed')
    st = st + opy.read(f'/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{info}.{end_day_j}.mseed')
    
    t1 = t0 + start_day*tdur - padding_hrs*3600
    t2 = t0 + end_day*tdur + padding_hrs*3600
    st_trim = st.copy().trim(starttime = t1, endtime = t2)
    st_trim.merge()
    return st_trim

def deconvolve_padded(st, padding_hrs=3, info=info):
    start_cut = st[0].stats.starttime + padding_hrs*3600
    end_cut = st[0].stats.endtime - padding_hrs*3600
    
    inventory = client.get_stations(network=netsel, station=stasel, channel=chnsel, location='*', starttime=st[0].stats.starttime, endtime=st[0].stats.endtime, level = 'response')
    st_decon = st.copy().remove_response(inventory = inventory)#removes response
    st_decon.trim(starttime = start_cut,endtime = end_cut)#trims to the days
    st_decon.detrend('demean')
    st_decon.write(f'/Volumes/LaCie/SN_Thesis/Deconvolved/{info}.{st[0].stats.starttime.julday+1}to{st[0].stats.endtime.julday-1}.Decon.mseed')
    
    return st_decon
    
st = create_padded_stream(60,90, 3)# will do days 0,1,2 with padding from -1 and 3
st.plot()
print('Done Creating Stream')
st_d = deconvolve_padded(st, 3)
st_d.plot()


def deconvolve_day(day):
    t1 = t0 + tdur*day #the start of the day we care about
    t2 = t0 + (day+1)*tdur#the end of the day we care about
    jday = t1.julday
    
    inventory = client.get_stations(network=netsel, station=stasel, channel=chnsel, location='*', starttime=t0, endtime=t0 + tdur, level = 'response')
    
    stream_raw_name = '/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{}.{}.mseed'.format(info,jday-1)
    stream_raw_name2 = '/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{}.{}.mseed'.format(info,jday)
    stream_raw_name3 = '/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{}.{}.mseed'.format(info,jday+1)
    
    print('Reading Streams')
    st_raw = opy.read(stream_raw_name)
    st_raw2 = opy.read(stream_raw_name2)
    st_raw3 = opy.read(stream_raw_name3)
    print('Adding Streams')
    st_raw = st_raw + st_raw2 + st_raw3 #add the three strings together
    
    print('Padding Streams')
    hour_padding = 3#hours to pad on each side
    pad_left =  t1- hour_padding*3600
    pad_right = t2 + hour_padding*3600
    st_raw.trim(starttime = pad_left , endtime = pad_right)
    
    print('Merging Streams')
    st_raw.merge()
    
    print('Deconvolving')
    st_decon = st_raw.copy().remove_response(inventory = inventory)
    
    st_decon.trim(starttime = t1,endtime = t2)
    st_decon.detrend('demean')
    
    decon_name = '/Volumes/LaCie/SN_Thesis/Deconvolved/{}.{}.Decon.mseed'.format(info,jday)
    st_decon.write(decon_name)
    print('Done')
    st_raw.plot()
    
    st_decon.plot()

    
end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )
    