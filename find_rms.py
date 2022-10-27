#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:18:33 2022

@author: brendanmills
"""
import obspy as opy
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import csv
import os

tdur=3600*24 # duration in seconds for one day
t0 = UTCDateTime("2018-06-26T0:0:00.000")
netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN07" # station code selection (* for all)
year = 2018
info = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)
client = Client("IRIS")

def calc_rms(stream, minutes, overlap):
    bin_size =minutes*60*100#min*seconds*samples
    tr = stream[0]
    start_day = tr.stats.starttime.julday
    end_day = tr.stats.endtime.julday

    rms = []
    time = []
    total_bins = len(tr)
    i = 0
    for t in np.arange(total_bins):
        if i + bin_size >= len(tr):
            break
        section = tr[i:(i+bin_size)]
        squared = section**2
        rms.append(np.sqrt(np.sum( squared )) / bin_size)
        t_now = t0 + i/100
        time.append(t_now)
        i += int(bin_size*(1-overlap))
    print('Done computing the RMS')
    
    #This sets up the data and formats it for the csv
    data_out = []
    data_out.append(time)
    data_out.append(rms)
    data_out = np.array(data_out)
    data_out = np.transpose(data_out)
    print('Done futzing with the array')
    
    #this part writes the csv
    file_name = f'/Volumes/LaCie/SN_Thesis/RMS/CSV/{info}-RMS{start_day}.{end_day}_bin{minutes}min_lap{overlap}.csv'
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_out)
    print(f'Done writing to the csv: {file_name}')
    
def get_decon_stream(start_day, end_day, info = info):
    try:
        stream = opy.read(f'/Volumes/LaCie/SN_Thesis/Decon_Ranges/{info}.{start_day}.{end_day}.Decon.mseed')
        return stream
    except:
        names = os.listdir('/Volumes/LaCie/SN_Thesis/Decon_Ranges/')
        selected = [] #we will only put the ones that have the correct station info in here
        for st in names:
            if st.startswith(info):
                selected.append(st)
        for st in selected:
            split_name = st.split('.')
            stream_start = int(split_name[5])
            stream_end = int(split_name[6])
            if start_day >= stream_start and start_day < stream_end:
                if end_day <= stream_end and end_day > stream_start:
                    stream = opy.read('/Volumes/LaCie/SN_Thesis/Decon_Ranges/' + st)
                    return stream
                    
            
                    
     
def get_rms(start_day, end_day, minutes, overlap):
    #remember there will be no data fron end_day
    t1 = t0 + start_day*tdur
    print(t1)
    time = np.zeros(0)
    data = np.zeros(0)
    
    file_name = f'/Volumes/LaCie/SN_Thesis/RMS/CSV/{info}-RMS{start_day}.{end_day}_bin{minutes}min_lap{overlap}.csv'
    try:#try reading the csv file if it already exists
        a = np.loadtxt(file_name, delimiter = ',',dtype = str)
    except:
        #we need to get the stream
        stream = opy.read(f'/Volumes/LaCie/SN_Thesis/Decon_Ranges/{info}.{start_day}.{end_day}.Decon.mseed')
        calc_rms(stream, minutes, overlap)
        a = np.loadtxt(file_name, delimiter = ',',dtype = str)
 
    for i in np.arange(len(a[:,0])) :
        time = np.append( time, UTCDateTime( a[i,0] ).datetime )
        try:
            data = np.append( data, (float( a[i,1] )) )
        except:
            data = np.append(data,0)

    return[time, data]