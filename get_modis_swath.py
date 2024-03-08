"""  """
#import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool
from pyhdf.SD import SD,SDC

from krttdkit.operate import enhance as enh
from krttdkit.acquire import modis
from krttdkit.acquire import laads

from FG1D import FG1D
import modis_rsrs

def modis_band_to_wl(band:int):
    """
    Returns the central wavelength of the integer band in um by finding
    the weighted mean wavelength of the spectral response.

    Spectral response functions provided by:
    https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_eos_1_modis_srf/
    """
    rsr_dict = modis_rsrs.modis_rsrs[band]
    wl = rsr_dict["wavelength"]
    rsr = rsr_dict["rsr"]
    mid_idx = np.argmin(np.abs(np.cumsum(np.array(rsr))-round(sum(rsr)/2)))
    return wl[mid_idx]

def get_modis_l1b(modis_l1b_file:Path, bands:tuple=None,
                   l1b_convert_reflectance:bool=True, l1b_convert_tb:bool=True,
                   debug=False):
    """
    Opens a Terra or Aqua L1b calibrated radiances or L2 atmospherically-
    corrected reflectance/emission granule file, and parses the
    requested bands based on keys defined in dictionaries above.

    The values returned by this method are contingent on the file type
    provided to datafile.

    For l1b files:
    sunsat and geolocation data is bilinear-interpolated from a 5x5 subsampled
    grid. By default reflectance bands are converted to BRDF and thermal
    bands are left as radiances.

    :@param modis_l1b_file: Path to level 1b 2 hdf4 file, probably from the
            laads daac.
    :@param bands: Band key defined in l2_products dictionary if datafile is
            a l2 path, or a valid MODIS band number.
    :return: (data, info, geo) 3-tuple. data is a list of ndarrays
            corresponding to each requested band, info is a list of data
            attribute dictionaries for the respective bands, and geo is a
            2-tuple (latitude, longitude) of the 1km data grid.
    """
    sd = SD(modis_l1b_file.as_posix(), SDC.READ)
    valid_bands = list(sorted(list(range(1,37)) + [13.5, 14.5]))
    if bands is None:
        bands = valid_bands
    else:
        assert all(b in valid_bands for b in bands)

    record_mapping = {
            "Band_250M":"EV_250_Aggr1km_RefSB",
            "Band_500M":"EV_500_Aggr1km_RefSB",
            "Band_1KM_Emissive":"EV_1KM_Emissive",
            "Band_1KM_RefSB":"EV_1KM_RefSB",
            }

    band_keys = {k:list(sd.select(k).get()) for k in record_mapping.keys()}
    labels = []
    data = []
    ctr_wls = []
    for b in bands:
        ctr_wls.append(modis_band_to_wl(int(b)))
        if b == 26:
            ## Band 26 is also in the normal 1km reflectance dataset,
            ## but my understanding is that one doesn't have the updated
            ## de-striping algorithm pre-applied.
            idx = band_keys["Band_1KM_RefSB"].index(26)
            tmp_sd = sd.select("EV_Band26")
            tmp_attrs = tmp_sd.attributes()
            tmp_data = tmp_sd.get()
        for k,band_list in band_keys.items():
            if b not in band_list:
                continue
            idx = band_list.index(b)
            tmp_sd = sd.select(record_mapping[k])
            tmp_attrs = tmp_sd.attributes()
            labels.append(b)
            tmp_data = tmp_sd.get()[idx]
        if "reflectance_units" in tmp_attrs.keys() \
                and l1b_convert_reflectance:
            ## Should be divided by cos(sza) for true BRDF
            tmp_data = (tmp_data-tmp_attrs["reflectance_offsets"][idx]) \
                    * tmp_attrs["reflectance_scales"][idx]
        elif not "reflectance_units" in tmp_attrs.keys() and l1b_convert_tb:
            c1 = 1.191042e8 # W / (m^2 sr um^-4)
            c2 = 1.4387752e4 # K um
            # Get brightness temp with planck's function at the ctr wl
            tmp_data = c2/(ctr_wls[-1]*np.log(c1/(ctr_wls[-1]**5*tmp_data)+1))
        else:
            tmp_data = (tmp_data-tmp_attrs["radiance_offsets"][idx]) \
                    * tmp_attrs["radiance_scales"][idx]
        data.append(tmp_data)
    data = np.stack(data, axis=-1)
    return labels,data,{"ctr_wls":ctr_wls}

def get_modis_geometry(modis_geoloc_file:Path, include_masks=False,):
    fields  = [
            ("Latitude", "lat"),
            ("Longitude", "lon"),
            ("Height", "height"),
            ("SensorZenith", "vza"),
            ("SensorAzimuth", "vaa"),
            ("SolarZenith", "sza"),
            ("SolarAzimuth", "saa"),
            ]
    if include_masks:
        fields += [
                ("Land/SeaMask", "m_land"),
                ("WaterPresent", "m_water"),
                ]
    sd = SD(modis_geoloc_file.as_posix(), SDC.READ)
    records,labels = zip(*fields)
    data = np.stack([sd.select(f).get() for f in records], axis=-1)
    return labels,data

def get_modis_swath(init_time:datetime, final_time:datetime, laads_token:str,
                    modis_nc_dir:Path, bands:tuple, isaqua=False,
                    keep_rad=False, keep_masks=False, ub_vza=180, ub_sza=180,
                    latlon_bbox=((-90,90),(-180,180)), adjsec=1200,
                    debug=False):
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
    geoloc_files = [
            laads.download(
                target_url=g["downloadsLink"],
                dest_dir=modis_nc_dir,
                raw_token=laads_token,
                debug=debug
                )
            for g in modis.query_modis_l1b(
                product_key=("MOD03","MYD03")[isaqua],
                start_time=init_time,
                end_time=final_time,
                debug=debug,
                )
            ]

    assert all(f.exists() for f in l1b_files)
    assert all(f.exists() for f in geoloc_files)
    dlabels,data,meta = zip(*[
        get_modis_l1b(
            modis_l1b_file=f,
            #bands=bands,
            #l1b_convert_reflectance=not keep_rad,
            #l1b_convert_tb=not keep_rad,
            #debug=debug
            )
        for f in l1b_files
        ])
    glabels,geom = zip(*[
            get_modis_geometry(modis_geoloc_file=f)
            for f in geoloc_files
            ])
    print(dlabels)
    print(glabels)
    print(data[0].shape)
    print(geom[0].shape)
    exit(0)

    #'''
    all_data = None
    labels = [I["band"] for I in modis_data[0][1]] + geo_sunsat_labels
    (lat0,latf),(lon0,lonf) = latlon_bbox
    for gran in modis_data:
        bands, info, geo, sunsat = gran
        tmp_data = np.dstack([*bands, *geo, *sunsat])
        in_range = np.logical_and(
                np.logical_and(
                    (tmp_data[..., labels.index("lat")] >= lat0),
                    (tmp_data[..., labels.index("lat")] < latf)
                    ),
                np.logical_and(
                    (tmp_data[..., labels.index("lon")] >= lon0),
                    (tmp_data[..., labels.index("lon")] < lonf)
                    )
                )
        in_range = np.logical_and(
                in_range,
                np.logical_and(
                    (tmp_data[..., labels.index("sza")]<=ub_sza),
                    (tmp_data[..., labels.index("vza")]<=ub_vza),
                    )
                )
        tmp_data = tmp_data[in_range]
        all_data = tmp_data if all_data is None \
                else np.concatenate((all_data,tmp_data))
    #'''
    return FG1D(labels, all_data)

def mp_get_modis_swath(swath:dict):
    """
    downloads MODIS data in a given time range
    """
    defaults = {
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
        raise e
        #print(e)
        return None



if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    modis_nc_dir = data_dir.joinpath("modis")
    modis_swath_dir = data_dir.joinpath("modis_swaths")

    """
    Generate a API download  token with an EarthData account here:
    https://ladsweb.modaps.eosdis.nasa.gov/profiles/#generate-token-modal

    Once you have it, put it directly into the 'laads-token.txt' file as
    raw text with no newline, ie `cat $TOKEN > token-dir/laads-token.txt`
    """
    token = str(data_dir.joinpath("laads-token.txt").open("r").read()).strip()

    """  --( configuration )--  """
    ## Specify a pickle file generated by get_ceres_swath.py,
    ## which should contain a list of FG1D-style tuples.
    swaths_pkl = data_dir.joinpath(
            #"ceres_swaths/ceres-ssf_hkh_terra_20180101-20201231.pkl")
            "ceres_swaths/ceres-ssf_hkh_aqua_20180101-20201231.pkl")
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
    #bbox = ((28,38), (-95,-75))
    workers = 1
    keep_netcdfs = False
    """  --( ------------- )--  """

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

    print(len(ceres_swaths))
    time_ranges = [time_ranges[0]]
    ceres_swaths = [ceres_swaths[0]]


    """
    Search for MODIS L1b files on the LAADS DAAC that were acquired at the same
    time and in the same area as each CERES swath identified by
    get_ceres_swath.py
    """
    ## download modis data near the footprints
    shared_args = {
            "laads_token":token,
            "modis_nc_dir":modis_nc_dir,
            "bands":modis_bands,
            "debug":debug

            ## Parse the following from swaths in multiprocessed method
            #"latlon_bbox":(
            #    ceres_swaths[0].meta.get("lat_range"),
            #    ceres_swaths[0].meta.get("lon_range")),
            ## Add 4 degrees for pixel buffer around outer footprints
            ## viewing zenith should be the limiting factor for the MODIS grid.
            #"ub_vza":ceres_swaths[0].meta.get("ub_vza") + 4,
            #"ub_sza":ceres_swaths[0].meta.get("ub_sza"),
            }

    ## Generate time ranges for each distinct ceres swath time range
    modis_args = [dict(
        shared_args,
        init_time=time_ranges[i][0],
        final_time=time_ranges[i][1],
        isaqua=ceres_swaths[i].meta.get("satellite") == "aqua",
        ) for i in range(len(time_ranges))]

    ##
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
                    "modis-swath" +
                    p_in["init_time"].strftime("_%Y%m%d-%H%M%S") +
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
