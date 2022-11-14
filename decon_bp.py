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

def gather_all():
    stream = opy.core.stream.Stream()
    for i in range(17):
        if i<10:
            sta = 'SN0'+str(i)
        else:
            sta = 'SN'+str(i)
        info = '8G.{}..{}.{}'.format(sta, chnsel, year)
        file_name = '{}.177.mseed'.format(info)
        try:
            info = '8G.{}..{}.{}'.format(sta, chnsel, year)
            file_name = '{}.177.mseed'.format(info)
            st = opy.read('/Volumes/LaCie/SIERRA_NEGRA/SN_GALAPAGOS/'+file_name)
            stream.append(st[0])
        except:
            pass
        try:
            info = 'EC.{}..{}.{}'.format(sta, chnsel, year)
            file_name = '{}.177.mseed'.format(info)
            st = opy.read('/Volumes/LaCie/SIERRA_NEGRA/SN_EC/'+file_name)
            stream.append(st[0])
        except:
            pass
    return stream

def deconvolve_prefilt(st, pf, out_name):
    st_decon = opy.core.stream.Stream()
    for tr in st.copy():
        stats = tr.stats
        inventory = client.get_stations(network=stats.network,
                                        station=stats.station, 
                                        channel=stats.channel, 
                                        location='*', 
                                        starttime=stats.starttime,
                                        endtime=stats.endtime, 
                                        level = 'response')
        
        st_d = tr.remove_response(inventory=inventory, pre_filt=pf, output='VEL')#removes response
        st_d.detrend('demean')
        st_decon.append(st_d)
        print('Done with ' + stats.station)
    file_name = '/Volumes/LaCie/SN_Thesis/Day177/'+out_name
    print(file_name)
    st_decon.write(file_name)
    return st_decon

stream = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.mseed')
files = ['8G.Array..HHZ.2018.177.BP1.mseed', 
         '8G.Array..HHZ.2018.177.BP2.mseed',
         '8G.Array..HHZ.2018.177.BP3.mseed',
         '8G.Array..HHZ.2018.177.BP4.mseed']
filters = [ [0.5, 0.7, 9, 10], 
            [0.5, 0.7, 17, 20],
            [0.001, 0.01, 17, 20],
            [0.001, 0.01, 40, 45]] 

stream = opy.read('/Volumes/LaCie/SN_Thesis/Day177/8G.Array..HHZ.2018.177.mseed')
for i in range(4):
    print('Start '+files[i])
    st_d = deconvolve_prefilt(stream, filters[i], files[i])
    
end_timer = time.time()
print('This all took {} seconds'.format( round(end_timer-start_timer,2)) )

# prefilt = [0.0001, 0.0002, samp_rate/2-2, samp]
# prefilt = [0.5, 0.7, 9, 10]
# st.remove_response(inventory=inventory, pre_filt= pre_filt, output = 'VEL', waterlevel= None)
# pre_filt1 = [0.5, 0.7, 9, 10]
# pre_filt2 = [0.5, 0.7, 17, 20]
# pre_filt3 = [0.001, 0.01, 40, 45]

