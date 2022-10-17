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

start_timer = time.time()
t0 = UTCDateTime("2018-06-26T0:0:00.000")
tdur=3600*24 # duration in seconds for one day

netsel="8G" # network code selection
chnsel="HHZ" # channel code selection
stasel="SN11" # station code selection (* for all)
year = 2018
info = '{}.{}..{}.{}'.format(netsel, stasel, chnsel, year)
client = Client("IRIS")

st = opy.read('/Volumes/LaCie/SN_Thesis/Decon_Ranges/8G.SN11..HHZ.2018.177.Decon.mseed')

