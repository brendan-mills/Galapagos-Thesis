#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:20:45 2023

@author: brendanmills
"""
import os
import pygmt
import numpy as np
import tqdm
import covseisnet as csn
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import obspy
from obspy import read_inventory, UTCDateTime
from obspy.clients.fdsn import mass_downloader
import rasterio
import scipy.interpolate
from glob import glob
from osgeo import gdal
import pandas as pd

depth_min = -1.5
depth_max = 5
sampling_rate = 25.0

win_duration_sec = 20
win_avg = 8
ovlp=0.5

low_pass = 1/win_duration_sec
high_pass = 12

ttmodel = '300lat20dep'
name = ttmodel + f'_w{win_duration_sec}a{win_avg}'
print(name)
PROJECT_PATH = '/Users/brendanmills/Documents/Senior_Thesis/Data/'
DIRPATH_RAW = PROJECT_PATH + 'Station_Data/Raw/'
PROCESSED_PATH = PROJECT_PATH + 'Station_Data/Processed/'
TTIMES_PATH = PROJECT_PATH + 'TTimes/' + ttmodel
META_PATH = PROJECT_PATH + 'Station_Data/Response/'
FIG_PATH = f'/Users/brendanmills/Documents/Senior_Thesis/Figs/Localization/{name}/'
BEAM_PATH = PROJECT_PATH + 'Beam/'
COVAR_PATH = PROJECT_PATH + f'Covariances/{name}/'
t0 = UTCDateTime("2018-06-26T17:0:00.000")

domain = mass_downloader.RectangularDomain(
    minlatitude=-1.17,#south
    maxlatitude=-0.45,#north
    minlongitude=-91.45,#west
    maxlongitude=-90.8,#east
)
# grid extent Longitude: 55.67° to 55.81° (145 points), Latitude: -21.3° to -21.2° (110 points)
lon_min = domain.minlongitude
lon_max = domain.maxlongitude
lat_min = domain.minlatitude
lat_max = domain.maxlatitude

#%% load in streams and preprocess
stream = csn.arraystream.read(PROCESSED_PATH + '8GEC.All..HHZ.Decon.filt1.trim.decim.mseed')
starttime = stream[0].stats.starttime
endtime = stream[0].stats.endtime
tdur = endtime - starttime

print('Preprocess')
stream.merge(method=1, fill_value="interpolate", interpolation_samples=-1)

## synchronize traces in the stream
stream = stream.synchronize(t0, tdur, method="linear")

## filtering
stream.detrend(type="demean")
stream.detrend(type="linear")
stream.filter(type="bandpass", freqmin=low_pass, freqmax=high_pass)

preproc_spectral_secs = win_duration_sec * win_avg * ovlp
stream.preprocess(
    domain="spectral", method="onebit", window_duration_sec=preproc_spectral_secs
)
stream.trim(starttime, endtime) 

#%% Calculate coherence
print('Covariance, Correlation and Beam')
os.makedirs(COVAR_PATH, exist_ok=True)
try:
    times = np.load(COVAR_PATH + 'times.npy')
    frequencies = np.load(COVAR_PATH + 'frequencies.npy')
    covariances = np.load(COVAR_PATH + 'covariances.npy').view(csn.CovarianceMatrix)
    print('Loaded Covariance')
except:
    times, frequencies, covariances = csn.covariancematrix.calculate(
        stream, win_duration_sec, win_avg
    )
    np.save(COVAR_PATH + 'times.npy', times)
    np.save(COVAR_PATH + 'frequencies.npy', frequencies)
    np.save(COVAR_PATH + 'covariances.npy', covariances)
print('First Eigen')
# Eigenvector decomposition - covariance matrix filtered by the 1st eigenvector to show the dominant source
try:
    covariance_1st = csn.covariancematrix.CovarianceMatrix(np.load(COVAR_PATH + 'cov1st.npy'))
    print('Loaded 1st Eigenvectors')
except:
    covariance_1st = covariances.eigenvectors(covariance=True, rank=0)
    np.save(COVAR_PATH + 'cov1st.npy', covariance_1st)
print('Calculating Correlation')
# Extract cross-correlations
lags, correlation = csn.correlationmatrix.cross_correlation(
    covariance_1st, sampling_rate
)
print('Beam from', TTIMES_PATH)
ttimes = csn.traveltime.TravelTime(stream, TTIMES_PATH)
# Initiate beam object and set geographical extent of grid
nwin = correlation.nwin()  # number of time windows
beam = csn.beam.Beam(nwin, ttimes)
beam.set_extent(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max)
# beam.set_extent(-91.3, -91, -0.9, -0.7, depth_min, depth_max)

#%% Location
try:
    df = pd.read_csv(FIG_PATH + 'max.csv')
    print('Loaded max.csv')
except:
    df = pd.DataFrame(columns=['long','lat','depth'])
    
    for i in tqdm.tqdm(range(nwin)):
        correl = correlation[i]
        # Filter correlation
        correl = correl.bandpass(low_pass, high_pass, sampling_rate)
    
        # Smooth correlation
        correl = correl.hilbert_envelope()
        correl = correl.smooth(sigma=20)  # default sigma is 5
    
        beam.calculate_likelihood(correl, sampling_rate, i)
    
        beam_max = beam.max_likelihood(i)
        df.loc[i] = [beam_max[0], beam_max[1], beam_max[2]]
    
    os.makedirs(FIG_PATH, exist_ok=True)
    print('Write to: ' + FIG_PATH + 'max.csv')
    df.to_csv(FIG_PATH + 'max.csv')


#%% Plotting
filepaths_raw = sorted(glob(os.path.join(DIRPATH_RAW, "*.mseed")))
filepaths_meta = sorted(glob(os.path.join(META_PATH,'*')))

inventory = read_inventory(META_PATH + '*xml')
# Extract stations
stations = [sta for net in inventory for sta in net]
attrs = "longitude", "latitude", "elevation", "code"
stations = [{item: getattr(sta, item) for item in attrs} for sta in stations]

stations = list(np.append(stations,[{'longitude': -91.409849, 'latitude': -0.791115, 'elevation': 46.796349, 'code': 'CEAZ'},
                  {'longitude': -91.01927, 'latitude': -0.8597631, 'elevation': 267.847, 'code': 'PVIL'},
                  {'longitude': -91.1134240, 'latitude': -0.7824234, 'elevation': 1067.3101983, 'code': 'VCH1'},
                  {'longitude': -90.9701114, 'latitude': -0.4548725, 'elevation': 134.2029101, 'code': 'ALCE'},
                  ]))

net = pd.DataFrame(stations)

stats = stream[0].stats

# Define region of interest around Sierra Negra
region = [-91.45, -90.8, -1.17, -0.45]#whole domain
region = [-91.25, -91, -0.9, -0.7]#zoomed in on caldera
# region = [-91.2, -91.09, -0.87, -0.75]#more zoom

# Load sample grid (3 arc-seconds global relief) in target area
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
grid = pygmt.grdclip(grid, below=[1, -2000])
# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

# --------------- plotting the hillshade map -----------

pygmt.makecpt(cmap="gray", series=[-1.5, 0.3, 0.01])
fig.grdimage(
    region=region,
    grid=dgrid,
    projection="M12c",
    frame=["SWrt+tTremor Location", "xa0.1", "ya0.1"],
    cmap=True,
    transparency=25,
)

track = pd.read_csv(FIG_PATH + 'max.csv')
#plot the path
dot_time = track.index/nwin*(endtime.hour-starttime.hour) + starttime.hour
pygmt.makecpt(cmap="viridis", series=[dot_time.min(), dot_time.max()])
fig.plot(x=track['long'],
          y=track['lat'],
          style="cc",
          size = 0.15*np.ones(len(track)),
          fill=dot_time,
          pen="black",
          cmap=True
          )
fig.colorbar(frame="af+lTime [hours UTC]")
fig.plot(x=net['longitude'],
          y=net['latitude'],
          style="t0.3t",
          # size = 0.15*np.ones(len(track)),
          fill='white',
          pen="black",
          )
#this plots te fissures
fissure_paths = sorted(glob(os.path.join(PROJECT_PATH + 'Fissures/', "*.txt")))
for file in fissure_paths:
    with open(file) as f:
        lines = np.array([line.strip().split() for line in f.readlines()])
        x = lines[:,0]
        y = lines[:,1]
        fig.plot(
            x=x,
            y=y,
            pen="2p,red",
        )
with open(PROJECT_PATH + 'Fissures/fissure_text.txt') as f:
    lines = np.array([line.strip().split() for line in f.readlines()])
    name = lines[:,0]
    x = lines[:,1]
    y = lines[:,2]
    fig.text(text=name, x=x, y=y)    

with fig.inset(
    position="jTR+o0.2c",
    box="+gwhite+p1p",
    region = [-92, -89, -1.5, 0.7],
    projection="M12/3c",
):
    # Highlight the Japan area in "lightbrown"
    # and draw its outline with a pen of "0.2p".
    fig.coast(land='lightgray',
              water='azure1'
    )
    # Plot a rectangle ("r") in the inset map to show the area of the main
    # figure. "+s" means that the first two columns are the longitude and
    # latitude of the bottom left corner of the rectangle, and the last two
    # columns the longitude and latitude of the uppper right corner.
    rectangle = [[region[0], region[2], region[1], region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="1.5p,blue")



fig.show()
fig.savefig(FIG_PATH + '/map.pdf')
#%% Array Figure
fig = pygmt.Figure()
region = [-91.8, -90.6, -1.07, 0.25]#whole domain
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=region)
# grid = pygmt.grdclip(grid, below=[1, -2000])
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
grid = pygmt.grdclip(grid, below=[1, -2000])
# calculate the reflection of a light source projecting from west to east
# (azimuth of 270 degrees) and at a latitude of 30 degrees from the horizon
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

pygmt.makecpt(cmap="gray", series=[-1.5, 0.3, 0.01])
fig.grdimage(
    region=region,
    grid=dgrid,
    projection="M12c",
    cmap=True,
    transparency=25,
)
fig.coast(
          water="azure1", 
          region=region,
          projection='M12c',
          frame=["SWrt+tStation Map", "xa0.2", "ya0.2"],
          )
stations = [sta for net in inventory for sta in net]
sta_temp = [{item: getattr(sta, item) for item in attrs} for sta in stations]

sta_perm = [{'longitude': -91.409849, 'latitude': -0.791115, 'elevation': 46.796349, 'code': 'CEAZ'},
                  {'longitude': -91.01927, 'latitude': -0.8597631, 'elevation': 267.847, 'code': 'PVIL'},
                  {'longitude': -91.1134240, 'latitude': -0.7824234, 'elevation': 1067.3101983, 'code': 'VCH1'},
                  {'longitude': -90.9701114, 'latitude': -0.4548725, 'elevation': 134.2029101, 'code': 'ALCE'},
                  ]
net_temp = pd.DataFrame(sta_temp)
net_perm = pd.DataFrame(sta_perm)
fig.plot(x=net_temp['longitude'],
          y=net_temp['latitude'],
          style='t0.3c',
          fill='green',
          pen="black",
          label='Temporary Stations'
          )
fig.plot(x=net_perm['longitude'],
          y=net_perm['latitude'],
          style='i0.3c',
          fill='red',
          pen="black",
          label='Permanent Stations'
          )

with fig.inset(position="jTR+w3c+o0.1c", margin=0):
    # Create a figure in the inset using coast. This example uses the azimuthal
    # orthogonal projection centered at 47E, 20S. The land color is set to
    # "gray" and Madagascar is highlighted in "red3".
    fig.coast(
        region="g",
        projection="G-91/-1/?",
        land="gray",
        water="white",
        dcw="MG+gred3",
    )
    fig.plot(
        x=-91, y=-45, style='v0.6c+e', direction=([90], [1]), pen="2p", fill="black"
    )

# fig.colorbar(frame=["a4000f2000", "x+lElevation", "y+lm"])
fig.legend(position="jTL+w3c+o0.1c")
fig.show() 
fig.savefig(PROJECT_PATH + '/stations.pdf')  
#%% Trash
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

