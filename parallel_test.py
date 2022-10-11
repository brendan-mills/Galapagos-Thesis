#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:37:45 2022

@author: brendanmills
"""

import multiprocessing
import time
import os
import obspy as opy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

start_timer = time.time()
t0 = UTCDateTime("2018-06-26T0:0:00.000")
tdur=3600*24 # duration in seconds for one day
client = Client("IRIS")
  
def decon_file(file):
    print(f'Begin {file}',flush=True)
    parts = file.split('.')#get the file information form the name
    net = parts[0]
    sta = parts[1]
    chan = parts[3]
    year = parts[4]
    day = parts[5]
    info = info = '{}.{}..{}.{}'.format(net, sta, chan, year)
    out = f'{info}.{day}.Decon.mseed'
    
    try:
        inventory = client.get_stations(network=net, station=sta, channel=chan, location='*', starttime=t0 + int(day)*tdur, endtime=t0 + (int(day)+1)*tdur, level = 'response')
    except:
        print(f'No response for {file}',flush=True)
    st = opy.read('/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/'+file)
    
    if bool(st.get_gaps()):
        print("This one has gaps",flush=True)
        out = f'{info}.{day}wGaps.Decon.mseed'
        
    if len(st[0]) == 1:
        print(f'This day is empty {file}',flush=True)
        return
        
    if out in os.listdir('/Volumes/LaCie/SN_Thesis/Parallel_Decon/'):
        print('Decon file already found',flush=True)
        return
        
    st_decon = st.copy().remove_response(inventory = inventory)
    out_name = f'/Volumes/LaCie/SN_Thesis/Parallel_Decon/{out}'
    st_decon.write(out_name)
    print(f'Wrote to {out_name}',flush=True)
   
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    inputs = os.listdir('/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/')
    outputs = pool.map(decon_file, inputs)
    

end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )