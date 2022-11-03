#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:11:26 2022

@author: brendanmills
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:05:49 2022

@author: brendanmills
"""
import obspy as opy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import time
#this is for looking at the onset tremor so the days are hardcoded for 177 and the times are relevant to the tremor
start_timer = time.time()
t0 = UTCDateTime("2018-06-26T0:0:00.000")
t1 = t0 + 14*3600#start of tremor
t2 = t0 + 21*3600#end of tremor
tdur=3600*24 # duration in seconds for one day

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN05" # station code selection (* for all)
year = 2018
info = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)
client = Client("IRIS")

def gather(sta_list):
#this will gather a stream with the stations we want
    stream = opy.core.stream.Stream()
    for sta in sta_list:
        info = '{}.{}..{}.{}'.format(netsel, sta, chnsel, year)
        file_name = '/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{}.177.mseed'.format(info)
        st = opy.read(file_name)
        stream.append(st[0])
    return stream

stas = ['SN04','SN05', 'SN07', 'SN11', 'SN12']

def deconvolve_prefilt(st):
    pre_filt = [0.5, 0.7, 9, 10]
    st_decon = opy.core.stream.Stream()
    for i in range(len(stas)):
        inventory = client.get_stations(network=netsel, station=stas[i], channel=chnsel, location='*', starttime=t0, endtime=t0+tdur, level = 'response')
        
        st_d = st[i].remove_response(inventory=inventory, pre_filt=pre_filt, output='VEL')#removes response
        st_d.detrend('demean')
        st_decon.append(st_d)
    file_name = '/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.BP1.Decon.mseed'
    print(file_name)
    st_decon.write(file_name)
    return st_decon

def gather_all():
    stream = opy.core.stream.Stream()
    for i in range(1,15):
        if i <10:
            sta = f'SN0{i+1}'
        else:
            sta = f'SN{i+1}'
        info = '{}.{}..{}.{}'.format(netsel, sta, chnsel, year)
        file_name = '/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/{}.177.mseed'.format(info)
        try:
            st = opy.read(file_name)
            stream.append(st[0])
        except:
            print(f'No data for {sta}')
    return stream

stream = gather(stas)
print(stream)
stream.plot()
st_d = deconvolve_prefilt(stream)
print(st_d)
st_d.plot()
end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )

# prefilt = [0.0001, 0.0002, samp_rate/2-2, samp]
# prefilt = [0.5, 0.7, 9, 10]
# st.remove_response(inventory=inventory, pre_filt= pre_filt, output = 'VEL', waterlevel= None)
# 
