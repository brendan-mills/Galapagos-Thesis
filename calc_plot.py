#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:20:45 2023

@author: brendanmills
"""
import cartopy
import os
import numpy as np
import tqdm
import covseisnet as csn
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import obspy
from obspy.clients.fdsn import Client
from obspy import read, read_inventory, UTCDateTime
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
from osgeo import gdal
from multiprocessing import Pool
import time


PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
TTIMES_PATH = PROJECT_PATH + 'TTimes/Sparse'
DEM_PATH = '/Users/brendanmills/Documents/Senior_Thesis/GalapagosDEM/'
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
t0 = t0 = UTCDateTime("2018-06-26T17:0:00.000")
tdur = 4*3600

domain = mass_downloader.RectangularDomain(
    minlatitude=-1,#south
    maxlatitude=-0.5794,#north
    minlongitude=-91.3508,#west
    maxlongitude=-90.9066,#east
)
# grid extent Longitude: 55.67° to 55.81° (145 points), Latitude: -21.3° to -21.2° (110 points)
lon_min = domain.minlongitude
lon_max = domain.maxlongitude
lat_min = domain.minlatitude
lat_max = domain.maxlatitude

depth_min = -1.5
depth_max = 20
sampling_rate = 25.0
# client = Client("IRIS")
beam_bank = []
#%% load in streams
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed")))
filepaths_meta = sorted(glob(os.path.join(DIRPATH_RAW, "*.xml")))
def get_streams(sta_array):
    stream = csn.arraystream.ArrayStream()
    for filepath_waveform in tqdm.tqdm(filepaths_raw, desc="Collecting Streams"):
        st = read(filepath_waveform)
        if st[0].stats.station in sta_array:
            stream.append(st[0]) 
    
    # decimate data to 25Hz
    stream.decimate(4)
    
    # download metadata
    inv = obspy.Inventory()
    for p in filepaths_meta:
        inv_to_be = read_inventory(p)
        if inv_to_be.get_contents()['channels'][0].split('.')[1] in sta_array:
            inv.extend(inv_to_be)
    return stream, inv


## merge traces to have one trace per station
def stream_pre_process(stream,low_pass, high_pass, win_dur_sec, avg, ovlp):
    print('Preprocess')
    stream.merge(method=1, fill_value="interpolate", interpolation_samples=-1)
    
    ## synchronize traces in the stream
    stream = stream.synchronize(t0, tdur, method="linear")
    
    ## filtering
    stream.detrend(type="demean")
    stream.detrend(type="linear")
    stream.filter(type="bandpass", freqmin=low_pass, freqmax=high_pass)
    
    preproc_spectral_secs = win_dur_sec * avg * ovlp
    stream.preprocess(
        domain="spectral", method="onebit", window_duration_sec=preproc_spectral_secs
    )
    stream.trim(t0, t0 + tdur)  # preprocessing can add artifacts to ends
    return stream

# Calculate coherence
def calc_correl(stream, win_dur_sec, avg):
    print('Correlation')
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, win_dur_sec, avg
    )
    
    # # Spectral width
    #spectral_width = covariances.coherence(kind="spectral_width")
    
    # # Average spectral width between 0.5Hz and 5Hz
    # i_freq_low = round(0.5 * spectral_width.shape[1] / sampling_rate)
    # i_freq_high = round(5 * spectral_width.shape[1] / sampling_rate)
    #spectral_width_average = np.mean(spectral_width[:, i_freq_low:i_freq_high], axis=1)
    
    # Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
    covariance_1st = covariances.eigenvectors(covariance=True, rank=0)
    
    # Extract cross-correlations
    lags, correlation = csn.correlationmatrix.cross_correlation(
        covariance_1st, sampling_rate
    )
    return correlation

# def beam_helper(i, corr_in, low_pass, high_pass, sigma, beam):
#     correl = corr_in[i]
#     correl = correl.bandpass(low_pass, high_pass, sampling_rate)
#     correl = correl.hilbert_envelope()
#     correl = correl.smooth(sigma=sigma)  # default sigma is 5

#     beam.calculate_likelihood(correl, sampling_rate, i)
    
#     beam_bank.append()
#     return f'Done with {i}'
    
# def parallel_beam(stream, corr_in, low_pass, high_pass, sigma=20):#default sig is 20
#     print('Beam')
#     ttimes = csn.traveltime.TravelTime(stream, TTIMES_PATH)
#     # Initiate beam object and set geographical extent of grid
#     nwin = corr_in.nwin()  # number of time windows
#     beam = csn.beam.Beam(nwin, ttimes)
#     beam.set_extent(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
#     items = [(i, corr_in, low_pass, high_pass, sigma, beam) for i in range(nwin)]
#     with Pool() as pool:
#         for out in pool.starmap(beam_helper, items):
#                 print(out)
#     return beam, nwin

def calc_beam(stream, corr_in, low_pass, high_pass, sigma=20):#default sig is 20
    print('Beam')
    ttimes = csn.traveltime.TravelTime(stream, TTIMES_PATH)
    # Initiate beam object and set geographical extent of grid
    nwin = corr_in.nwin()  # number of time windows
    beam = csn.beam.Beam(nwin, ttimes)
    beam.set_extent(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
    
    # Loop through all windows and calculate likelihood 
    for i in range(0, nwin):
        print("Processing window", i + 1, "of", nwin)
    
        correl = corr_in[i]
    
        # Filter correlation
        correl = correl.bandpass(low_pass, high_pass, sampling_rate)
    
        # Smooth correlation
        correl = correl.hilbert_envelope()
        correl = correl.smooth(sigma=sigma)  # default sigma is 5
    
        beam.calculate_likelihood(correl, sampling_rate, i)

        beam_max = beam.max_likelihood(i)
        print(
            "Maximum likelihood at",
            round(beam_max[0], 4),
            "\N{DEGREE SIGN},",
            round(beam_max[1], 4),
            "\N{DEGREE SIGN},",
            round(beam_max[2], 1),
            "km",
        )
    return beam, nwin

def sw_insert(stream_in, sw_in):
    # # plot nrf and spectral width and waveforms from closest station
    duration_min = tdur / 60
    fig, ax = plt.subplots(2, constrained_layout=True, figsize=(10, 8))  # stretched plot
    
    trace_plot = stream_in.select(station="SN07", channel="HHZ")[0].data
    ax[0].plot(np.linspace(0, duration_min, len(trace_plot)), trace_plot, "k")
    ax[0].set_title("Vertical channel of station FOR")
    ax[0].set_ylabel("FOR.HHZ (counts)")
    ax[0].set_xlim([0, duration_min])
    
    img = ax[1].imshow(
        sw_in.T,
        origin="lower",
        cmap="jet_r",
        interpolation="none",
        extent=[0, tdur / 60, 0, sampling_rate],
        aspect="auto",
    )
    ax[1].set_ylim(
        [0.5, stream_in[0].stats.sampling_rate / 2]
    )  # hide microseismic background noise below 0.5Hz
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_yscale("log")
    ax[1].set_title("Spectral Width")
    plt.colorbar(img, ax=ax[1]).set_label("Covariance matrix spectral width")
    
    # # Average spectral width between 0.5Hz and 5Hz
    # i_freq_low = round(0.5 * sw_in.shape[1] / sampling_rate)
    # i_freq_high = round(5 * sw_in.shape[1] / sampling_rate)
    #spectral_width_average = np.mean(sw_in[:, i_freq_low:i_freq_high], axis=1)
    
    # ax[2].plot(np.linspace(0, duration_min, nwin), spectral_width_average, "k")
    # ax[2].set_title("Average spectral width between 0.5Hz and 5Hz")
    # ax[2].set_ylabel("Spectral width")
    # ax[2].set_xlim(0, duration_min)
    # ax[2].set_xlabel("Minutes")
    

def get_field(beam, i):
    print('Getting field')
    # extract max likelihood position
    #beam_max = beam.max_likelihood(i_win)
    
    # Choose the last window for plotting likelihood
    likelihood_xyz = beam.likelihood[i, :, :, :]

    # Take slices at point of max likelihood
    i_max, j_max, k_max = np.unravel_index(likelihood_xyz.argmax(), likelihood_xyz.shape)
    likelihood_xy = likelihood_xyz[:, :, k_max]
    
    # _max = np.amax(beam.likelihood)
    # _min = np.min(beam.likelihood)
    
    # likelihood_xz = likelihood_xyz[:, j_max]
    # likelihood_yz = likelihood_xyz[i_max]
    
    # Normalize likelihood between 0 and 1
    likelihood_xy = (likelihood_xy - likelihood_xy.min()) / (
        likelihood_xy.max() - likelihood_xy.min()
    )
    # likelihood_xz = (likelihood_xz - likelihood_xz.min()) / (
    #     likelihood_xz.max() - likelihood_xz.min()
    # )
    # likelihood_yz = (likelihood_yz - likelihood_yz.min()) / (
    #     likelihood_yz.max() - likelihood_yz.min()
    # )
    # # Normalize likelihood between 0 and 1 my way
    # likelihood_xy = (likelihood_xy - _min) / (
    #     _max - _min
    # )
    return likelihood_xy

def dem_prep(ttimes):
    print('DEM Prep')
    # clip the dem to the bounds using gdal
    dataset = gdal.Open(DEM_PATH + 'GalDEMclip.tif')
    
    # path to where you want the clipped raster
    outputSrtm = DEM_PATH + 'GalDEMclipzoom.tif'
    gdal.Translate(outputSrtm , dataset,projWin = [lon_min, lat_max, lon_max, lat_min])
    
    # Download DEM and interpolate to grid
    dem_path = outputSrtm
    dem = rasterio.open(dem_path)  # open downloaded dem file
    dem1 = dem.read(1)  # extract values
    dem1 = np.where(dem1 == -999, 0, dem1)  # replace null values with zero
    nx_dem = dem1.shape[0]  # x dimension of dem grid
    ny_dem = dem1.shape[1]  # y dimension of dem grid
    # old dem grid
    x,y = np.mgrid[0 : nx_dem - 1 : complex(nx_dem), 0 : ny_dem - 1 : complex(ny_dem)]
    # new dem grid, with dimensions matching our traveltime grid
    x2, y2 = np.mgrid[
        0 : nx_dem - 1 : complex(ttimes.nx),
        0 : ny_dem - 1 : complex(ttimes.ny),
    ]
    # interpolate onto the new grid
    dem2 = scipy.interpolate.griddata(
        (x.ravel(),y.ravel()), dem1.ravel(), (x2, y2), method="linear"
    )
    
    # create custome discrete colormap for topography
    levels = 12
    custom_cmap_dem = plt.cm.get_cmap("Greys")(np.linspace(0.2, 0.5, levels))
    custom_cmap_dem = mcolors.LinearSegmentedColormap.from_list("Greys", custom_cmap_dem)
    
    # prepare shaded topography
    # create light source object.
    ls = LightSource(azdeg=315, altdeg=45)
    # shade data, creating an rgb array.
    rgb = ls.shade(dem2, custom_cmap_dem)
    np.save(DEM_PATH  + 'rgb.npy',rgb)
    return
def my_cmap():
    # create custom discrete colormap for likelihood
    low = 4
    levels = 12
    n_colours = 16  # number of discrete colours to split colourbar into
    custom_cmap = plt.cm.get_cmap("RdYlBu_r")(np.linspace(0, 1, levels))
    for i in range(low):
        custom_cmap[i, :] = [1, 1, 1, 1]
    for i in range(1, levels):
        custom_cmap[i, -1] = np.sqrt(i / levels)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "RdYlBu_r", custom_cmap, N=n_colours
    )
    return custom_cmap

def plot_map(field, win_dur, stream, inv, i, run_name):
    print('Mapping')
    rgb = np.load(DEM_PATH+'rgb.npy')
    a_cmap = my_cmap()
    fig, ax = plt.subplots(1, constrained_layout=True, dpi=100)
    img_xy = ax.imshow(
        field.T,
        interpolation="none",
        origin="lower",
        cmap=a_cmap,
        #cmap=plt.get_cmap("Oranges"),
        aspect="auto",
        extent=[lon_min, lon_max, lat_min, lat_max],
        vmin=0.5,
        vmax=1,
    )
    ax.imshow(
        rgb,
        interpolation="none",
        alpha=0.35,
        cmap=plt.get_cmap("Greys"),
        aspect="auto",
        extent=[lon_min, lon_max, lat_min, lat_max],
    )
    # ax.add_patch(
    #     plt.Circle((x_max, y_max), facecolor="black", edgecolor="white", radius=0.001)
    # )  # plot max likelihood position
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    itime = t0 + win_dur * i
    ax.set_title(f"{run_name}, map view, {itime}")
    
    
    os.makedirs(FIG_PATH + run_name + '/', exist_ok=True)
    
    fig.savefig(FIG_PATH+ f'{run_name}/SNi{i}')
    
    # create dictionary of station metadata from station xml
    net = {"lat": [], "lon": []}
    for tr in stream:
        inv_sel = inv.select(station=tr.stats.station)
        net["lat"].append(inv_sel[0][0].latitude)
        net["lon"].append(inv_sel[0][0].longitude)
    for x, y in zip(net["lon"], net["lat"]):  # plot stations
        triangle = RegularPolygon(
            (x, y), facecolor="white", edgecolor="black", numVertices=3, radius=0.0045
        )
        ax.add_patch(triangle)
    plt.colorbar(img_xy).set_label("Likelihood")
    return fig, ax

def corr_and_plot(low, high, win, av, ov, sta_array, name):
    # (low, high, win, av, ov, sta_array, name) = big_in
    print(low, high, win, av, ov, sta_array, name)
    st, inv = get_streams(sta_array)
    print(st)
    stream_out = stream_pre_process(st, low, high, win, av, ov)
    correlation = calc_correl(stream_out, win, av)
    beam, nwin = calc_beam(stream_out, correlation,low, high, sigma=20)
    for i in range(nwin):
        field = get_field(beam, i)
        plot_map(field, tdur/nwin,stream_out, inv, i, name)
    return name
        
# def par_test(low, high, win, av, ov, stream, name):
#     stream_out = stream_pre_process(stream, low, high, win, av, ov)
#     correlation = calc_correl(stream_out, win, av)
#     beam, nwin = parallel_beam(stream_out, correlation,low, high, sigma=20)
#     for i in range(nwin):
#         field = get_field(beam, i)
#         plot_map(field, tdur/nwin, i, name)


# corr_and_plot(0.5, 10, 120, 30, 0.5, stream, 'test')

# par_test(0.5, 10, 120, 30, 0.5, stream, 'ParallelTest')
# staa = ['SN04', 'SN05','SN07','SN11','SN12','SN13','SN14','SN06']
# corr_and_plot( (6, 8, 120, 30, 0.5, staa, 'test') )


#%% Parameters

# # frequency limits for filtering (depends on the target signal)
# low_pass = 0.5
# high_pass = 10.0

# # optimized for VT earthquakes
# # window_duration_sec = 12
# # average = 20

# # optimized for tremors
# window_duration_sec = 180
# average = 30
# overlap = 0.5

# stream_out = stream_pre_process(stream, low_pass, high_pass, window_duration_sec, average, overlap)
# correlation = calc_correl(stream_out, window_duration_sec, average)
# beam, nwin = calc_beam(stream_out, correlation,sigma=20)
# for i in range(nwin):
#     field, small, big = get_field(beam, i)
#     plot_map(field,small, big, tdur/nwin, i, 'test')

#%% for depth plots
# dem_x = -1 * dem2[i_max, :] / 1000  # dem along xz slice, convert to km
# dem_y = np.flip(-1 * dem2[:, j_max] / 1000)  # dem along yz slice, convert to km
'''
Run0 - 0.5, 10, 60,30,0.5
Run1 - 0.5, 10, 60,30,0.75
Run2 - 0.5, 10, 60,10,0.5
Run3 - 0.5, 10, 120,30,0.5
Run4 - 0.5, 10, 180,30,0.5

'''