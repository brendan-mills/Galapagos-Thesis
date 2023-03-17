#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:43:29 2023

@author: brendanmills
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
#These are the OGs
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
runs = ['pRun1/','pRun2/','pRun3/','pRun4/','pRun5/','pRun6/']
os.makedirs(FIG_PATH + 'Batch1/', exist_ok=True)
for j in range(14):
    fig = plt.figure()
    
    for i in range(len(runs)):
        img = Image.open(FIG_PATH + runs[i] + 'SNi' + str(j) + '.png')
        fig.add_subplot(2, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(FIG_PATH + 'Batch1/Frame' + str(j), dpi = 200)

#%% These exclude 13
runs = ['pRun8/','pRun9/','pRun10/','pRun10/','pRun12/','pRun13/']
os.makedirs(FIG_PATH + 'Batch2/', exist_ok=True)
for j in range(14):
    fig = plt.figure()
    
    for i in range(len(runs)):
        img = Image.open(FIG_PATH + runs[i] + 'SNi' + str(j) + '.png')
        fig.add_subplot(2, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(FIG_PATH + 'Batch2/Frame' + str(j), dpi = 200)

#%% These exclude 5 and 11
runs = ['pRun15/','pRun16/','pRun17/','pRun18/','pRun19/','pRun20/']
os.makedirs(FIG_PATH + 'Batch3/', exist_ok=True)
for j in range(14):
    fig = plt.figure()
    
    for i in range(len(runs)):
        img = Image.open(FIG_PATH + runs[i] + 'SNi' + str(j) + '.png')
        fig.add_subplot(2, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(FIG_PATH + 'Batch3/Frame' + str(j), dpi = 200)
