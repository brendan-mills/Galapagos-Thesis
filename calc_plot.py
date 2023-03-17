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
import pandas as pd


PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Raw/'
DIRPATH_PROCESSED = PROJECT_PATH + 'Processed/'
TTIMES_PATH = PROJECT_PATH + 'TTimes/Sparse5'
DEM_PATH = '/Users/brendanmills/Documents/Senior_Thesis/GalapagosDEM/'
FIG_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/'
BEAM_PATH = PROJECT_PATH + 'Beam/'
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
depth_max = 5
sampling_rate = 25.0
# client = Client("IRIS")
#%% load in streams
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed")))
filepaths_meta = sorted(glob(os.path.join(DIRPATH_RAW, "*.xml")))
def get_streams(sta_array, decimate=True):
    stream = csn.arraystream.ArrayStream()
    for filepath_waveform in tqdm.tqdm(filepaths_raw, desc="Collecting Streams"):
        st = read(filepath_waveform)
        if st[0].stats.station in sta_array:
            stream.append(st[0]) 
    
    # decimate data to 25Hz
    if decimate:
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
    # print('times', times.shape)
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
    # print('correlation shape', correlation.shape)
    return correlation

def calc_beam(stream, corr_in, low_pass, high_pass, sigma=20, test=False):#default sig is 20
    print('Beam from', TTIMES_PATH)
    ttimes = csn.traveltime.TravelTime(stream, TTIMES_PATH)
    # Initiate beam object and set geographical extent of grid
    nwin = corr_in.nwin()  # number of time windows
    beam = csn.beam.Beam(nwin, ttimes)
    beam.set_extent(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
    # Loop through all windows and calculate likelihood 
    df = pd.DataFrame(columns=['long','lat','depth'])
    
    for i in tqdm.tqdm(range(0, nwin)):
        # print("Processing window", i + 1, "of", nwin)
    
        correl = corr_in[i]
    
        # Filter correlation
        correl = correl.bandpass(low_pass, high_pass, sampling_rate)
    
        # Smooth correlation
        correl = correl.hilbert_envelope()
        correl = correl.smooth(sigma=sigma)  # default sigma is 5
    
        beam.calculate_likelihood(correl, sampling_rate, i)

        beam_max = beam.max_likelihood(i)
        # print(
        #     i,
        #     'Max at',
        #     round(beam_max[0], 4),
        #     "\N{DEGREE SIGN},",
        #     round(beam_max[1], 4),
        #     "\N{DEGREE SIGN},",
        #     round(beam_max[2], 1),
        #     "km,",
        #     round(np.mean(beam.likelihood[i]),5),
        #     round(np.std(beam.likelihood[i]),5)      
        # )
        df.loc[len(df.index)] = [beam_max[0], beam_max[1], beam_max[2]]
        if test:
            break
    return beam, df

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
    
    # Average spectral width between 0.5Hz and 5Hz
    i_freq_low = round(0.5 * sw_in.shape[1] / sampling_rate)
    i_freq_high = round(5 * sw_in.shape[1] / sampling_rate)
    spectral_width_average = np.mean(sw_in[:, i_freq_low:i_freq_high], axis=1)

def norm_field(field, norm, big, small):
    likelihood_xyz = field
    _max = big
    _min = small
    # Take slices at point of max likelihood
    i_max, j_max, k_max = np.unravel_index(likelihood_xyz.argmax(), likelihood_xyz.shape)
    likelihood_xy = likelihood_xyz[:, :, k_max]
    
    likelihood_xz = likelihood_xyz[:, j_max]
    likelihood_yz = likelihood_xyz[i_max]
    
    # Normalize likelihood frame by frame between 0 and 1
    if norm=='frame':
        likelihood_xy = (likelihood_xy - likelihood_xy.min()) / (
            likelihood_xy.max() - likelihood_xy.min()
        )
        likelihood_xz = (likelihood_xz - likelihood_xz.min()) / (
            likelihood_xz.max() - likelihood_xz.min()
        )
        likelihood_yz = (likelihood_yz - likelihood_yz.min()) / (
            likelihood_yz.max() - likelihood_yz.min()
        )
    elif norm=='log':
        likelihood_xy = (likelihood_xy - _min) / (
        _max - _min
        )
        likelihood_xz = (likelihood_xz - _min) / (
            _max - _min
        )
        likelihood_yz = (likelihood_yz - _min) / (
            _max - _min
        )
        bump = 1.5
        likelihood_xy = np.log(likelihood_xy + bump)
        likelihood_xz = np.log(likelihood_xz + bump)
        likelihood_yz = np.log(likelihood_yz + bump)
    elif norm=='my-way':
        # Normalize likelihood between 0 and 1 my way
        likelihood_xy = (likelihood_xy - _min) / (
        _max - _min
        )
        likelihood_xz = (likelihood_xz - _min) / (
            _max - _min
        )
        likelihood_yz = (likelihood_yz - _min) / (
            _max - _min
        )
    elif norm == 'no-way':
        pass
    else:
        print('That is not a valid normalization method')
    return likelihood_xy, likelihood_xz, likelihood_yz

def get_field(beam, i, norm):
    # print('Getting field')
    # extract max likelihood position
    #beam_max = beam.max_likelihood(i_win)
    
    # Choose the last window for plotting likelihood
    likelihood_xyz = beam.likelihood[i, :, :, :]
    _max = np.amax(beam.likelihood)
    _min = np.amin(beam.likelihood)
    return norm_field(likelihood_xyz, norm, _max, _min)
    
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
    np.save(DEM_PATH  + 'dem2.npy',dem2)
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

def dem_extras(beam, i_win):
    dem2 = np.load(DEM_PATH + 'dem2.npy')
    i_max, j_max, k_max = np.unravel_index(beam.likelihood[i_win, :, :, :].argmax(), beam.likelihood[i_win, :, :, :].shape)
    dem_x = -1 * dem2[i_max, :] / 1000  # dem along xz slice, convert to km
    dem_y = np.flip(-1 * dem2[:, j_max] / 1000)  # dem along yz slice, convert to km
    return dem_x, dem_y

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
    # print('Mapping')
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
    ax.set_title(f"{run_name}, map view, {i} {itime}")
    
    
    os.makedirs(FIG_PATH + run_name + '/', exist_ok=True)
    
    
    
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
    fig.savefig(FIG_PATH+ f'{run_name}/SNi{i}')
    return fig, ax

def plot_depth(xz, yz, i, run_name, beam):
    # plot depth views
    fig = plt.figure(figsize=(12, 7))
    a_cmap = my_cmap()
    dem_x, dem_y = dem_extras(beam, i)
    ax1 = fig.add_subplot(2,1,1)
    
    img_xz = ax1.imshow(
        xz.T,
        interpolation="none",
        origin="upper",
        cmap=a_cmap,
        aspect="auto",
        extent=[lon_min, lon_max, depth_max, depth_min],
        vmin=0.5,
    )
    ax1.fill_between(
        np.linspace(lon_min, lon_max, len(dem_x)),
        depth_min,
        dem_x,
        facecolor="w",
        edgecolor="k",
        lw=0.4,
    )  # crop out data above topo
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title("Likelihood location, depth view")
    fig.colorbar(img_xz).set_label("Likelihood")
    
    ax2 = fig.add_subplot(2,1,2)
    img_yz = ax2.imshow(
        yz.T,
        interpolation="none",
        origin="upper",
        cmap=a_cmap,
        aspect="auto",
        extent=[lat_min, lat_max, depth_max, depth_min],
        vmin=0.5,
    )
    ax2.fill_between(
        np.linspace(lat_min, lat_max, len(dem_y)),
        depth_min,
        dem_y,
        facecolor="w",
        edgecolor="k",
        lw=0.4,
    )  # crop out data above topo
    ax2.set_xlabel("Latitude")
    ax2.set_ylabel("Depth (km)")
    fig.colorbar(img_yz).set_label("Likelihood")
    os.makedirs(FIG_PATH + run_name + '/Depth', exist_ok=True)
    fig.savefig(FIG_PATH+ f'{run_name}/Depth/SNi{i}')
    return


    

def corr_and_plot(low, high, win, av, ov, sta_array, name, norm, test=False):
    os.makedirs(FIG_PATH + name, exist_ok=True)
    print(low, high, win, av, ov, sta_array, name)
    st, inv = get_streams(sta_array)
    print(st)
    stream_out = stream_pre_process(st, low, high, win, av, ov)
    correlation = calc_correl(stream_out, win, av)
    nwin = correlation.nwin()
    beam, df = calc_beam(stream_out, correlation,low, high, sigma=20, test=test)
    df.to_csv(FIG_PATH + name + '/max.csv')
    with open(FIG_PATH + name +'/' + 'info.txt', 'w') as f:
        f.write(f'Nwin {nwin}\n')
        f.write(f'Run {name}\n')
        f.write(f'Low {low}\n')
        f.write(f'High {high}\n')
        f.write(f'Window_sec {win}\n')
        f.write(f'Avgerage {av}\n')
        f.write(f'Overlap {ov}\n')
        f.write(f'Stations {str(sta_array)}\n')
        f.write(f'Norm {norm}\n')
    for i in range(nwin):
        fieldxy, xz, yz = get_field(beam, i, norm)
        plot_map(fieldxy, tdur/nwin,stream_out, inv, i, name)
        try:
            
            plot_depth(xz, yz, i, name, beam)
        except:
            with open(FIG_PATH + name +'/' + 'info.txt', 'w') as f:
                f.write('I screwed up the depth figure\n')
            
    return name

'''
Run0 - 0.5, 10, 60,30,0.5
Run1 - 0.5, 10, 60,30,0.75
Run2 - 0.5, 10, 60,10,0.5
Run3 - 0.5, 10, 120,30,0.5
Run4 - 0.5, 10, 180,30,0.5

'''