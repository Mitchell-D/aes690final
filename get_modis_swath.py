"""
"""
import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool

from krttdkit.operate import enhance as enh
from krttdkit.products import FeatureGrid
#from krttdkit.products import HyperGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.acquire import modis
from krttdkit.acquire import laads

from FG1D import FG1D

def get_modis_swath(init_time:datetime, final_time:datetime, laads_token:str,
                    modis_nc_dir:Path, bands:tuple, latlon_bbox:tuple,
                    keep_rad=False, adjsec=1200, isaqua=False, debug=False):
    """
    Kicks off process of developing datasets of MODIS pixels that are clustered
    alongside nearby CERES footprints. This includes downloading all MODIS L1b
    files in range, opening and subsetting them to the appropriate size,
    and extracting an ordered sequence of imager pixels that are in range of
    the CERES footprints.

    1. Downloads 1 or more MOD021KM granules
    2. Parses out the requested bands within the latlon range
    3. returns a FG1D object with all in-range pixels.

    :@param adjsec: "adjecency seconds", or the time window in seconds within
        which times are assumed to be part of the swath. This is a sanity check
        threshold making sure files from different overpasses don't download.
    """
    geo_sunsat_labels = ["lat", "lon", "height", "sza", "saa", "vza", "vaa"]
    #t0,tf = init_time.timestamp(), final_time.timestamp()
    l1b_files = [
            laads.download(
                target_url=g["downloadsLink"],
                dest_dir=modis_nc_dir,
                raw_token=laads_token,
                debug=debug
                )
            for g in modis.query_modis_l1b(
                product_key=("MOD021KM","MYD021KM")[isaqua],
                start_time=init_time,
                end_time=final_time,
                debug=debug,
                )
            ]
    assert all(f.exists() for f in l1b_files)
    modis_data = [
            modis.get_modis_data(
                datafile=f,
                bands=bands,
                l1b_convert_reflectance=not keep_rad,
                l1b_convert_tb=not keep_rad,
                debug=debug
                )
            for f in l1b_files
            ]
    all_data = None
    labels = None
    lats,lons = tuple(latlon_bbox)
    lat0,latf = tuple(lats)
    lon0,lonf = tuple(lons)
    for gran in modis_data:
        bands, info, geo, sunsat = gran
        tmp_data = np.dstack([*bands, *geo, *sunsat])
        labels = [I["band"] for I in info] + geo_sunsat_labels \
                if labels is None else labels
        in_range = np.logical_and(
                np.logical_and(
                    (tmp_data[:,:,labels.index("lat")] >= lat0),
                    (tmp_data[:,:,labels.index("lat")] < latf)
                    ),
                np.logical_and(
                    (tmp_data[:,:,labels.index("lon")] >= lon0),
                    (tmp_data[:,:,labels.index("lon")] < lonf)
                    )
                )
        tmp_data = tmp_data[in_range]
        all_data = tmp_data if all_data is None \
                else np.concatenate((all_data,tmp_data))
    return FG1D(labels, [all_data[:,i] for i in range(len(labels))])

def mp_get_modis_swath(swath:dict):
    """
    downloads MODIS data in a given time range
    """
    defaults = {
            "latlon_bbox":((-90,90),(-180,180)),
            "isaqua":False,
            "debug":False,
            }
    args = dict(defaults, **swath)
    mandatory_args = ("init_time","final_time","laads_token",
                      "modis_nc_dir", "bands")
    try:
        assert all(k in args.keys() for k in mandatory_args)
        return args,get_modis_swath(**args)
    except Exception as e:
        #raise e
        print(e)
        return None



if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    modis_nc_dir = Path("/rstor/mdodson/modis_seus")
    modis_swath_dir = data_dir.joinpath("modis")
    """
    Generate your own token with an EarthData account here:
    https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal

    Once you have it, put it directly into the 'laads-token' file as
    raw text with no newline, ie `cat $TOKEN > token-dir/laads-token`
    """
    token = str(data_dir.joinpath("laads-token").open("r").read()).strip()

    #swaths_pkl = data_dir.joinpath("buffer/terra_ceres_seus_2017.pkl")
    #swaths_pkl = data_dir.joinpath("buffer/aqua_ceres_seus_2017.pkl")

    #swaths_pkl = data_dir.joinpath("buffer/terra_ceres_seus_2015.pkl")
    #swaths_pkl = data_dir.joinpath("buffer/aqua_ceres_seus_2015.pkl")

    #swaths_pkl = data_dir.joinpath("buffer/terra_ceres_seus_2021.pkl")
    #swaths_pkl = data_dir.joinpath("buffer/aqua_ceres_seus_2021.pkl")

    #swaths_pkl = data_dir.joinpath("buffer/terra_ceres_seus_2019.pkl")
    swaths_pkl = data_dir.joinpath("buffer/aqua_ceres_seus_2019.pkl")

    modis_bands = [
            8,              # .41                       Near UV
            1,4,3,          # .64,.55,.46               (R,G,B)
            2,              # .86                       NIR
            18,             # .935                      NIR water vapor
            5,26,6,7,       # 1.24, 1.38, 1.64, 2.105   SWIR (+cirrus)
            20,             # 3.7                       Magic IR
            27,28,          # 6.5, 7.1                  (high,low) layer pwv
            30,             # 9.7                       ozone
            31,             # 10.9                      clean window
            33,             # 13.3                      co2
            ]
    ## lat,lon preset for seus
    bbox = ((28,38), (-95,-75))
    workers = 20
    keep_netcdfs = False

    """
    Use the epoch time bounds from a CERES swath to set limits
    on the acquisition times of MODIS granules.
    """
    ceres_swaths = [FG1D(*s) for s in pkl.load(swaths_pkl.open("rb"))]
    ceres_labels = ceres_swaths[0].labels
    assert all(set(C.labels)==set(ceres_labels) for C in ceres_swaths)
    ## Get the time ranges from the individual footprint epoch times
    time_ranges = [(datetime.fromtimestamp(np.min(C.data("epoch"))),
                    datetime.fromtimestamp(np.max(C.data("epoch"))))
                   for C in ceres_swaths]
    #ceres = FG1D(ceres_labels, np.concatenate([C.data for C in ceres_swaths]))

    """ Use pool multiprocessing to """
    ## download modis data near the footprints
    isaqua = "aqua" in swaths_pkl.name.lower()
    assert isaqua or "terra" in swaths_pkl.name.lower()
    shared_args = {
            "laads_token":token,
            "modis_nc_dir":modis_nc_dir,
            "bands":modis_bands,
            "latlon_bbox":bbox,
            "isaqua":isaqua,
            "debug":debug
            }
    ## Generate time ranges for each distinct ceres swath time range
    modis_args = [dict(shared_args, init_time=t0, final_time=tf)
                  for t0,tf in time_ranges]
    with Pool(workers) as pool:
        swath_count = 0
        for p_all in pool.imap(mp_get_modis_swath, modis_args):
            trange = time_ranges[swath_count]
            ceres = ceres_swaths[swath_count]
            if p_all is None:
                continue
            p_in,p_out = p_all
            ## Construct output pkl name
            tmp_file = modis_swath_dir.joinpath(
                    ("terra","aqua")[isaqua] +
                    p_in["init_time"].strftime("_%Y%m%d-%H%M%S") +
                    p_in["final_time"].strftime("_%Y%m%d-%H%M%S") +
                    ".pkl"
                    )
            try:
                ## Store both the 1D CERES and the 1D MODIS data in the pkl.
                assert not tmp_file.exists()
                pkl.dump((p_out.to_tuple(), ceres.to_tuple()),
                         tmp_file.open("wb"))
            except Exception as e:
                #raise e
                print(e)
            finally:
                swath_count += 1
