"""
This script extracts CERES footprints from valid terra/aqua passes
over a region, and separates them into 1D lists of labeled features
corresponding to each independent swath.

These are stored in the swaths_pkl configured below, which is
formatted as a list of 2-tuples corresponding to F string labels of
features and a (D,F) shaped array of D data points (footprints).

The list entries are separated into flyovers with varying numbers
of valid footprints. Passes with too few valid footprints are
thresholded by min_footprints

swaths = [ FG1D(labels,data) for labels,data in swaths ]
"""
from pprint import pprint as ppt
import json
import netCDF4 as nc
import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import linregress
#from sklearn.linear_model import LinearRegression

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from krttdkit.acquire import modis
from krttdkit.operate import enhance as enh
#from krttdkit.products import FeatureGrid

from FG1D import FG1D

def load_modis_l2(mod09_path:Path, bands:list):
    data, info, geo = modis.get_modis_data(
            datafile=mod09_path,
            bands=bands,
            )
    return FeatureGrid(labels=bands, data=data, info=info)

def get_world_latlon(res=.1):
    return np.meshgrid(np.arange(-90,90,res),
                       np.arange(-180,180,res),
                       indexing="ij")

def print_ceres(ceres_nc:Path):
    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    for k in ds.variables.keys():
        X = ds.variables[k][:]
        stat = enh.array_stat(X)
        print(f"{str(X.shape):<12} {k:<70} {stat['min']:.3f} {stat['max']:.3f}")

def parse_ceres(ceres_nc:Path):
    """
    Parses fields from a full-featured CERES SSF file
    """
    label_mapping = [
        ## (M,) time and geometry information
        ("Time_of_observation", "jday"),
        ("lat", "lat"),
        ("lon", "lon"),
        ("CERES_viewing_zenith_at_surface", "vza"),
        ("CERES_relative_azimuth_at_surface", "raa"),
        ("CERES_solar_zenith_at_surface", "sza"),

        ## (M,8) Most prominent surface types, in decreasing order
        ("Surface_type_index",
         ("id_s1","id_s2","id_s3","id_s4",
          "id_s5","id_s6","id_s7","id_s8")),
        ("Surface_type_percent_coverage",
         ("pct_s1","pct_s2","pct_s3","pct_s4",
          "pct_s5","pct_s6","pct_s7","pct_s8")),
        ("Clear_layer_overlap_percent_coverages",
         ("pct_clr","pct_l1","pct_l2","pct_ol")),

        ## (M,) ADM-corrected fluxes
        ("CERES_SW_TOA_flux___upwards", "swflux"),
        ("CERES_WN_TOA_flux___upwards", "wnflux"),
        ("CERES_LW_TOA_flux___upwards", "lwflux"),

        ("Cloud_mask_clear_strong_percent_coverage", "nocld"),
        ("Cloud_mask_clear_weak_percent_coverage", "nocld_wk"),

        ## (M,2) COD for each layer weighted by PSF and cloud fraction
        ("Mean_visible_optical_depth_for_cloud_layer",
         ("l1_cod","l2_cod")),
        ("Stddev_of_visible_optical_depth_for_cloud_layer",
         ("l1_sdcod","l2_sdcod")),

        ## (M,) PSF weighted percentage of pixels in the footprint which
        ## have either land or ocean aerosol values
        ("Percentage_of_CERES_FOV_with_MODIS_land_aerosol", "aer_land_pct"),
        ## (M,) PSF weighted cloud frac from MOD04:
        ## Cloud fraction from Land aerosol cloud mask from retrieved
        ## and overcast pixels not including cirrus mask
        ("PSF_wtd_MOD04_cloud_fraction_land", "aer_land_cfrac"),
        ## (M,) Weighted integer percentage bins of aerosol types
        ("PSF_wtd_MOD04_aerosol_types_land", "aer_land_type"),
        ("PSF_wtd_MOD04_corrected_optical_depth_land__0_550_", "aod_land"),

        ## (M,) Optical depth with the deep blue method (?)
        ("Percentage_of_CERES_FOV_with_MODIS_deep_blue_aerosol", "aer_db_pct"),
        ("PSF_wtd_MOD04_deep_blue_aerosol_optical_depth_land__0_550_", "aod_db"),

        ## (M,) Over-ocean aerosol properties
        ("Percentage_of_CERES_FOV_with_MODIS_ocean_aerosol", "aer_ocean_pct"),
        ("PSF_wtd_MOD04_cloud_fraction_ocean", "aer_ocean_cfrac"),
        ("PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_", "aod_ocean"),
        ("PSF_wtd_MOD04_optical_depth_small_average_ocean__0_550_", "aod_ocean_small"),
        ]

    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    #unq_ids = np.ma.unique(ds.variables["Surface_type_index"][:])
    #print(f"Valid IDs: {unq_ids}")
    for ncl,l in label_mapping:
        X = ds.variables[ncl][:]
        if not type(l) is str:
            assert len(l) == X.shape[1]
            for i in range(len(l)):
                data.append(X[:,i])
                labels.append(l[i])
        else:
            assert len(X.shape)==1
            data.append(X)
            labels.append(l)
    return labels, data

def interp_modis(M, C, lbl="swflux"):
    """
    Load a L2 (atmospherically corrected) file with geolocation, make a plot
    with the provided label
    """
    #print(fg.shape, fg.labels)
    lon = C.data("lon")
    lat = C.data("lat")
    req_data = C.data(lbl)

    """ NN-Interpolate shortwave fluxes onto the 1km grid """
    interp = NearestNDInterpolator(list(zip(lon,lat)),req_data)
    regridded = interp(M.data("longitude"), M.data("latitude"))
    plt.pcolormesh(M.data("longitude"), M.data("latitude"), regridded)
    plt.show()
    return None

def contour_plot(hcoords, vcoords, data, bins, plot_spec={}):
    ps = {"xlabel":"latitude", "ylabel":"longitude", "marker_size":4,
          "cmap":"nipy_spectral", "text_size":12, "title":"",
          "norm":"linear","figsize":(12,12)}
    ps.update(plot_spec)
    fig,ax = plt.subplots()
    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))

    cont = ax.contourf(hcoords, vcoords, data, bins, cmap=ps.get("cmap"))
    fig.colorbar(cont)
    plt.show()

def interp_ceres(C, lbl="swflux", plot_spec={}):
    """ Interpolate the onto a regular grid """
    clat = np.arange(30,45,.1)
    clon = np.arange(-135,-120,.1)
    lat = C.data("lat")
    lon = C.data("lon")
    d = C.data(lbl)
    regrid = griddata((lat, lon), d, (clat[:,None], clon[None,:]),
                     method="nearest")
    contour_plot(clon,clat,regrid, 50, plot_spec=plot_spec)
    return None

def jday_to_epoch(jday:float, ref_jday:int, ref_gday:datetime):
    """
    Given a reference jday given as an integer, coresponding to noon on the
    provided reference gregorian day, and a float offset in decimal days
    with respect to the reference datetime, returns a gregorian datetime.

    This is genuinely probably the best way to do this with the ceres ssf data.
    """
    return (ref_gday + timedelta(days=jday-ref_jday)).timestamp()

if __name__=="__main__":
    labels = [ ## custom label mappings for CERES bands (encoded above)
            'jday', 'lat', 'lon', 'vza', 'raa', 'sza',
            'id_s1', 'id_s2', 'id_s3', 'id_s4',
            'id_s5', 'id_s6', 'id_s7', 'id_s8',
            'pct_s1', 'pct_s2', 'pct_s3', 'pct_s4',
            'pct_s5', 'pct_s6', 'pct_s7', 'pct_s8',
            'pct_clr', 'pct_l1', 'pct_l2', 'pct_ol',
            'swflux', 'wnflux', 'lwflux',
            'nocld', 'nocld_wk', 'l1_cod', 'l2_cod', 'l1_sdcod', 'l2_sdcod',
            'aer_land_pct', 'aer_land_cfrac', 'aer_land_type', 'aod_land',
            'aer_db_pct', 'aod_db', 'aer_ocean_pct', 'aer_ocean_cfrac',
            'aod_ocean', 'aod_ocean_small'
            ]

    data_dir = Path("data")
    ## Upper bound on sza to restrict daytime pixels
    ub_sza = 80
    ## Upper bound on viewing zenith angle (MODIS FOV is like 45)
    ub_vza = 35
    ## minimum amount of time between swaths (sec)
    lb_swath_interval = 300
    ## maximum amount of time included in a swath (sec)
    ub_swath_interval = lb_swath_interval*5
    ## Minimum number of valid footprints that warrant storing a swath
    min_footprints = 50
    ## The easiest way to convert from julian is wrt a specific reference day.
    ref_jday = 2444239.5
    ref_gday = datetime(year=1980,month=1,day=1)

    #sy = [(sat,year) for year in (2015,2017,2021) for sat in ("terra", "aqua")]
    #ceres_years = (2017, 2021)
    #sy = [(sat,year) for year in ceres_years for sat in ("terra", "aqua")]

    ceres_files = list(data_dir.joinpath("ceres").iterdir())
    #for sat,year in sy:
    for cf in ceres_files:
        #t0,tf = datetime(year,1,1),datetime(year,12,31,23,59,59,999999)
        #ceres_file = data_dir.joinpath(
        #        f"ceres/ceres-ssf_e4a_{sat}_{year}0101-{1+year}0101.nc")
        #swaths_pkl = data_dir.joinpath(f"buffer/{sat}_ceres_seus_{year}.pkl")

        print_ceres(cf)

        continue

        ## process only one CERES order
        ceres = FG1D(*parse_ceres(cf))

        '''
        """ Load combined CERES granules """
        ## Combine all ceres bands in the list
        ceres = FG1D(labels, list(map(np.concatenate, zip(*tuple(
            data for _,data in map(parse_ceres, ceres_files)
            )))))
        '''

        '''
        """ buffered labels are extracted into a pkl for quicker access """
        #buf_pkl = data_dir.joinpath("buffer/")
        buf_labels = ["jday", "lat", "lon", "vza", "raa", "sza",
                      "swflux", "wnflux", "lwflux"]
        '''
        '''
        ## Write a pickle containing a subset of the bands for convenience.
        pkl.dump([ceres.data(l) for l in buf_labels], buf_pkl.open("wb"))
        ## Load a pickle containing a subset of the bands.
        buf_data = pkl.load(buf_pkl.open("rb"))
        ceres = FG1D(buf_labels, buf_data)
        '''

        print(f"footprints, features : {ceres.data().shape}")
        print(f"unique days: {np.unique(np.round(ceres.data('jday'))).size}")

        """ Swap julian calendar for epoch seconds """
        epoch = np.array([jday_to_epoch(jday, ref_jday, ref_gday)
                          for jday in ceres.data("jday")])
        ceres.drop_data("jday")
        ceres.add_data("epoch", epoch)

        ceres = ceres.mask(np.logical_and(
            (ceres.data("epoch")>=t0.timestamp()),
            (ceres.data("epoch")<tf.timestamp())
            ))
        cday = ceres.mask(ceres.data("sza")<=ub_sza)

        ## Only consider daytime swaths for now.
        ceres = cday.mask(cday.data("vza")<=ub_vza)

        ## More than 5 minutes probably means it's a different swath.
        all_swaths = []
        tmask = np.concatenate((np.array([True]), np.diff(
            ceres.data("epoch"))>lb_swath_interval))
        approx_times = ceres.data("epoch")[tmask]
        for stime in approx_times:
            ## Look for anything with a stime offset less than 15min
            smask = np.abs(ceres.data("epoch")-stime)<lb_swath_interval*3
            swath = ceres.mask(smask)
            all_swaths.append(swath)

        ## Keep all swaths with at least 50 footprints in range.
        pkl.dump([s.to_tuple() for s in all_swaths if s.size>min_footprints],
                 swaths_pkl.open("wb"))
