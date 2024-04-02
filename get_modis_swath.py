"""  """
#import gc
import random
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool
from collections import ChainMap

## Need to use pyhdf for hdf4 and h5py for hdf5 :(
from pyhdf.SD import SD,SDC
import h5py

from krttdkit.operate import enhance as enh
from krttdkit.acquire import modis
from krttdkit.acquire import laads

from FG1D import FG1D
from FeatureGridV2 import FeatureGridV2 as FG
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

def get_modis_l1b(modis_l1b_file:Path, bands:tuple=None, keep_rad:bool=False,
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
        if b == 26:
            ## Band 26 is also in the normal 1km reflectance dataset,
            ## but my understanding is that one doesn't have the updated
            ## de-striping algorithm pre-applied.
            idx = 0 #band_keys["Band_1KM_RefSB"].index(26)
            tmp_sd = sd.select("EV_Band26")
            keys = ["reflectance_offsets", "reflectance_scales",
                    "radiance_offsets", "radiance_scales",
                    "reflectance_units"]
            tmp_attrs = {k:[tmp_sd.attributes()[k]] for k in keys}
            raw_data = tmp_sd.get()
        else:
            for k,band_list in band_keys.items():
                if b not in band_list:
                    continue
                idx = band_list.index(b)
                tmp_sd = sd.select(record_mapping[k])
                tmp_attrs = tmp_sd.attributes()
                raw_data = tmp_sd.get()[idx]
                break
        if "reflectance_units" in tmp_attrs.keys() \
                and l1b_convert_reflectance:
            ## Should be divided by cos(sza) for true BRDF
            #print(tmp_attrs["reflectance_offsets"])
            #print(b)
            tmp_data = (raw_data-tmp_attrs["reflectance_offsets"][idx]) \
                    * tmp_attrs["reflectance_scales"][idx]
        elif not "reflectance_units" in tmp_attrs.keys() and l1b_convert_tb:
            c1 = 1.191042e8 # W / (m^2 sr um^-4)
            c2 = 1.4387752e4 # K um
            # Get brightness temp with planck's function at the ctr wl
            tmp_data = c2/(ctr_wls[-1]*np.log(c1/(ctr_wls[-1]**5*raw_data)+1))
        labels.append(b)
        data.append(tmp_data)
        ctr_wls.append(modis_band_to_wl(int(b)))
        if keep_rad:
            tmp_data = (raw_data-tmp_attrs["radiance_offsets"][idx]) \
                    * tmp_attrs["radiance_scales"][idx]
            labels.append(f"{b}-rad")
            data.append(tmp_data)
    data = np.stack(data, axis=-1)
    return labels,data,{"ctr_wls":ctr_wls}

def get_modis_geometry(modis_geoloc_file:Path, include_masks:bool=False,
        include_geod_vectors:bool=False):
    """
    Expected input file follows the MxD03 standard:
    https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MOD03

    :@param modis_geoloc_file: MxD03 Geolocation netCDF (from LAADS DAAC)
    :@param include_masks: If True,
    """
    fields  = [
            ("Latitude", "lat"),
            ("Longitude", "lon"),
            ("Height", "height"),
            ("SensorZenith", "vza"),
            ("SensorAzimuth", "vaa"),
            ("SolarZenith", "sza"),
            ("SolarAzimuth", "saa"),
            ]
    ## Viewing angles are in degrees*10
    scale = np.array([1., 1., 1., .01, .01, .01, .01])
    if include_masks:
        fields += [
                ("Land/SeaMask", "m_land"),
                ("WaterPresent", "m_water"),
                ]
        ## Add unit scale values for the masks
        scale = np.concatenate([scale, np.array([1,1])])
    ## Open the file and read the requested fields in order
    sd = SD(modis_geoloc_file.as_posix(), SDC.READ)
    records,labels = zip(*fields)
    ## Stack all of the data to a (M,N,G) grid for G geometric features
    data = np.stack([
        sd.select(f).get().astype(float)/scale
        for f in records
        ], axis=-1)
    ## If requested, calculate geodetic X,Y,Z vectors to each pixel
    if include_geod_vectors:
        colat,lon = data[...,0],90.-data[...,1]
        v_img = np.stack([
            np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
            np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
            np.cos(np.deg2rad(lat))
            ])

    return labels,data

def get_modis_swath(ceres_swath:FG1D, laads_token:str, modis_nc_dir:Path,
                    swath_h5_dir:Path, region_label:str,
                    lat_buffer:float=0., lon_buffer:float=0.,
                    bands:tuple=None, isaqua=False, keep_rad=False,
                    keep_masks=False, spatial_chunks:tuple=(64,64),
                    buf_size_mb=128, debug=False):
    """
    Kicks off process of developing datasets of MODIS pixels that are clustered
    alongside nearby CERES footprints. This includes downloading all MODIS L1b
    files in range, opening and subsetting them to the appropriate size,
    and extracting an ordered sequence of imager pixels that are in range of
    the CERES footprints.

    1. Downloads 1 or more MOD021KM granules
    2. Parses out the requested bands within the latlon range
    3. returns a FG1D object with all in-range pixels.

    """
    geo_sunsat_labels = ["lat", "lon", "height", "sza", "saa", "vza", "vaa"]
    #t0,tf = init_time.timestamp(), final_time.timestamp()

    init_time = datetime.fromtimestamp(np.min(ceres_swath.data("epoch")))
    final_time = datetime.fromtimestamp(np.max(ceres_swath.data("epoch")))
    avg_time = datetime.fromtimestamp(np.average(ceres_swath.data("epoch")))

    ## Extract a grid that is slightly larger than the footprint bounds.
    latlon_bbox=(
            (np.amin(ceres_swath.data("lat"))-lat_buffer,
             np.amax(ceres_swath.data("lat"))+lat_buffer),
            (np.amin(ceres_swath.data("lon"))-lat_buffer,
             np.amax(ceres_swath.data("lon"))+lat_buffer),
            )
    ## Add 2 degrees for pixel buffer around outer footprints
    ## viewing zenith should be the limiting factor for the MODIS grid.
    isaqua = ceres_swath.meta.get("satellite") == "aqua"
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
            bands=bands,
            keep_rad=keep_rad,
            debug=debug,
            )
        for f in l1b_files
        ])

    glabels,geom = zip(*[
            get_modis_geometry(
                modis_geoloc_file=f,
                include_masks=keep_masks,
                include_geod_vectors=True,
                )
            for f in geoloc_files
            ])

    print(f"Swath count: {len(data)}")
    ## Zonally concatenate overpasses so that North is up.
    ## Terra is descending during the day, and aqua is ascending.
    data = np.concatenate(data, axis=0)[::(-1,1)[isaqua]]
    geom = np.concatenate(geom, axis=0)[::(-1,1)[isaqua]]
    ## Concatenate data and geometric features along the feature axis
    data = np.concatenate((data,geom), axis=-1)
    print(geom)
    print(f"Data shape: {data.shape}")

    ## Since aqua is ascending during the day, its scan is inverted.
    if isaqua:
        data = data[::-1,::-1]

    modis_fg = FG(
            clabels=("y","x"),
            flabels=list(dlabels[0])+list(glabels[0]),
            data=data,
            meta=dict(ChainMap(*meta)),
            )

    m_lat = (modis_fg.data("lat") >= latlon_bbox[0][0]) \
            & (modis_fg.data("lat") < latlon_bbox[0][1])
    m_lon = (modis_fg.data("lon") >= latlon_bbox[1][0]) \
            & (modis_fg.data("lon") < latlon_bbox[1][1])

    r_lat = np.squeeze(np.array(np.where(np.any((m_lat & m_lon), axis=1))))
    r_lon = np.squeeze(np.array(np.where(np.any((m_lat & m_lon), axis=0))))
    r_lat = np.amin(r_lat), np.amax(r_lat)
    r_lon = np.amin(r_lon), np.amax(r_lon)

    if debug:
        print(f"before",modis_fg.shape)
    modis_fg = modis_fg.subgrid(y=r_lat, x=r_lon)
    if debug:
        print(f"after",modis_fg.shape)

    timestr = avg_time.strftime(f"%Y%m%d-%H%M")
    ## Open file for writing with 128MB buffer
    sat = ceres_swath.meta.get("satellite")
    h5_path = swath_h5_dir.joinpath(
            f"swath_{region_label}_{sat}_{timestr}.h5")
    f_swath = h5py.File(h5_path, "w-", rdcc_nbytes=buf_size_mb*1024**2)
    g_swath = f_swath.create_group("/data")
    g_swath.attrs["modis"] = modis_fg.to_json()
    g_swath.attrs["ceres"] = ceres_swath.to_json()
    ## Not worth chunking ceres datasets, which shouldn't be enormous
    d_ceres = g_swath.create_dataset(
            name="ceres",
            shape=ceres_swath.data().shape,
            compression="gzip"
            )
    d_modis = g_swath.create_dataset(
            name="modis",
            shape=modis_fg.shape,
            chunks=(*spatial_chunks,modis_fg.shape[-1]),
            compression="gzip",
            )
    d_ceres[...] = ceres_swath.data()
    d_modis[...] = modis_fg.data()
    f_swath.close()
    return h5_path

def mp_get_modis_swath(swath:dict):
    """
    downloads MODIS data in a given time range
    """
    defaults = {"debug":False}
    args = dict(defaults, **swath)
    mandatory_args = ("ceres_swath","laads_token", "modis_nc_dir")
    try:
        assert all(k in args.keys() for k in mandatory_args)
        return args,get_modis_swath(**args)
    except Exception as e:
        raise e
        #print(e)
        return None



if __name__=="__main__":
    debug = False
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
    #swaths_pkl = data_dir.joinpath(
    swaths_pkls = [
            "ceres_swaths/ceres-ssf_idn_aqua_20180101-20200916_0mod3.pkl",
            "ceres_swaths/ceres-ssf_azn_aqua_20180101-20201231_0mod3.pkl",
            "ceres_swaths/ceres-ssf_hkh_aqua_20180101-20201231_0mod3.pkl",
            "ceres_swaths/ceres-ssf_neus_aqua_20180101-20201129_0mod3.pkl",
            #"ceres_swaths/ceres-ssf_idn_aqua_20200916-20201231.pkl"
            #"ceres_swaths/ceres-ssf_idn_terra_20180101-20200815.pkl"
            #"ceres_swaths/ceres-ssf_idn_terra_20200816-20201231.pkl"
            ]
    swaths_pkls = list(map(lambda p: data_dir.joinpath(p), swaths_pkls))
    '''
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_hkh_terra_20180101-20201231.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_hkh_aqua_20180101-20201231.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_azn_terra_20180101-20201231.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_azn_aqua_20180101-20201231.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_neus_terra_20180101-20201112.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_neus_terra_20201112-20201231.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_neus_aqua_20180101-20201129.pkl"),
            data_dir.joinpath(
                "ceres_swaths/ceres-ssf_neus_aqua_20201129-20201231.pkl"),
            ]
    '''
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
    workers = 5
    keep_netcdfs = False
    """  --( ------------- )--  """

    ## Extract and shuffle all the listed swaths together;
    ## beware memory constraints here.
    ceres_swaths = []
    for p in swaths_pkls:
        ceres_swaths += [FG1D(*s) for s in pkl.load(p.open("rb"))]
    random.shuffle(ceres_swaths)

    ceres_swaths = ceres_swaths[:50]

    """
    Search for MODIS L1b files on the LAADS DAAC that were acquired at the same
    time and in the same area as each CERES swath from get_ceres_swath.py
    """
    ## download modis data near the footprints
    shared_args = {
            "laads_token":token,
            ## Directory where MODIS netCDF files are deposited
            "modis_nc_dir":modis_nc_dir,
            "swath_h5_dir":modis_swath_dir,
            ## MODIS bands to extract. None for all bands.
            "bands":None,
            ## Extract the MODIS grid out to a couple degrees latitude and
            ## longitude outside the minimum and maximum footprint centroid
            ## boundaries. This is needed in order to extract grids around
            ## outer centroids.
            "lat_buffer":2,
            "lon_buffer":2,
            ## Size of hdf5 chunks in the first 2 dimensions of the hdf5
            ## (64,64,32) chunks of 8 byte float equates to 1MB per chunk
            "spatial_chunks":(64,64),
            ## Chunk buffer volume in MB
            "buf_size_mb":512,
            "debug":debug,
            }


    ## Generate time ranges for each distinct ceres swath time range
    modis_args = [
            dict(shared_args, ceres_swath=s, region_label=s.meta.get("region"))
            for s in ceres_swaths
            ]
    with Pool(workers) as pool:
        swath_count = 0
        for args,f in pool.map(mp_get_modis_swath, modis_args):
            print(f"Generated swath file: {f.as_posix()}")
