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
#from krttdkit.products import HyperGrid

from FG1D import FG1D

def haversine(lat1, lon1, lat2, lon2, height=6.3781e6):
    """
    Calculate great circle distance of equal length arguments.
    This method is intended to be numpy-vectorized
    """
    lat1,lon1,lat2,lon2 = map(np.radians, (lat1,lon1,lat2,lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    return height * 2 * np.arcsin(np.sqrt(a))

def aggregate_ceres_modis(swath_path:tuple, keep_closest:int,
        modis_features:list=None, debug=False):
    """
    1. Load imager pixel data from modis pkl
    2. Iterate over footprints, calculating great circle distance to each
       pixel and keeping up to the closest K pixels.
    3. convert the geometry of K pixels nearest each fooprint to
       azimuth/radius coordinates wrt the footprint lat/lon.
    4. Discard modis pixels' geodetic and angular coordinates.
       Consider keeping each MODIS pixel sun/sat geometry in the future.
    5. Return a (N,1,C) shaped array of N ceres footprints with C ceres
       features and a (N,K,M) array of M modis bands for up to K pixels
       nearby the N footprints.
    """
    valid_ceres = [
            "lat", "lon", "vza", "raa", "sza", "swflux", "lwflux", "epoch"
            ]
    modis,ceres = map(lambda d:FG1D(*d), pkl.load(swath_path.open("rb")))
    ceres = ceres.subset(valid_ceres)

    if debug:
        print(f"Extracting {ceres.size} footprints from {swath_path.name}")
    M = np.vstack(modis._data).T
    C = np.vstack(ceres._data).T
    modis_idx_lat = modis.labels.index("lat")
    modis_idx_lon = modis.labels.index("lon")
    ceres_idx_lat = ceres.labels.index("lat")
    ceres_idx_lon = ceres.labels.index("lon")
    mlat, mlon = modis.data("lat"),modis.data("lon")
    footprints = []
    nancount = 0
    for i in range(C.shape[0]):
        ## Get the current ceres footprint
        cpx = C[i]
        ## Get the great circle distance
        clat, clon = cpx[ceres_idx_lat],cpx[ceres_idx_lon]
        gcdist = haversine(clat, clon, modis.data("lat"), modis.data("lon"))

        ## Rank pixels by great circle proximity
        dist_order = gcdist.argsort()

        ## Extract the closest pixels to the footprint
        mpx = M[dist_order[:keep_closest]]

        '''
        """ Don't deal with null values """
        if not np.logical_and(np.all(np.isfinite(mpx)),
                np.all(np.isfinite(cpx))):
            nancount += 1
            continue
        '''

        mlat,mlon = mpx[:,modis_idx_lat],mpx[:,modis_idx_lon]
        gcdpx = gcdist[dist_order[:keep_closest]]
        #print([f"{g:.2f}" for g in gcdpx[:5]]) ## distance sanity check

        ## Get the azimuth relative to the ceres reference
        dlon = np.deg2rad(clat-mlat)
        y = np.sin(dlon) * np.cos(np.deg2rad(clat))
        x = np.cos(np.deg2rad(mlat))*np.sin(np.deg2rad(clat)) - \
                np.sin(np.deg2rad(mlat))*np.cos(np.deg2rad(clat))*np.cos(dlon)
        azi = np.arctan2(y,x)

        ## Collect modis pixels and append great circle distance and azimuth
        ## for this footprint. still includes modis lat/lon.
        mpx = np.hstack([mpx, gcdpx[:,None], azi[:,None]])[None]
        footprints.append((cpx[None,None], mpx))

        '''
        for l,m in [(modis.labels[j],np.average(mpx[:,j]))
                for j in range(len(modis.labels))]:
            print(l,m)
        print()
        for l,c in [(ceres.labels[j],cpx[j])
                for j in range(len(ceres.labels))]:
            print(l,c)
        '''
    return tuple(zip((ceres.labels, modis.labels+["dist", "azi"]),
        map(np.vstack, zip(*footprints))))

def mp_aggregate_ceres_modis(args:tuple):
    """
    given a 2-tuple like (ceres:FG1D, modis:Path), aggregates modis
    pixels around the reported geodetic coordinates of ceres footprints
    """
    try:
        return aggregate_ceres_modis(**args)
    except Exception as e:
        raise e
        #print(e)
        return None

def parse_modis_dt(file_name:Path):
    """
    parses start and end time of MODIS swath pkls, which are formatted
    like %Y%m%d-%H%M%S in the last 2 underscore-separated fields.

    :@return: (start:datetime, end:datetime)
    """
    return tuple(map(
        lambda s: datetime.strptime(s,"%Y%m%d-%H%M%S"),
        file_name.stem.split("_")[-2:]
        ))

if __name__=="__main__":
    '''
    """ Not used here; just included for reference """
    ##                      # MODIS Band Selection (in order)
    modis_bands = [
            8,              # .41                       Near UV
            1,4,3,          # .64, .55, .46             (R,G,B)
            2,              # .86                       NIR
            18,             # .935                      NIR water vapor
            5,26,6,7,       # 1.24, 1.38, 1.64, 2.105   SWIR (+cirrus)
            20,             # 3.7                       Magic IR
            27,28,          # 6.5, 7.1                  (high,low) layer pwv
            30,             # 9.7                       ozone
            31,             # 10.9                      clean window
            33,             # 13.3                      co2
            ]               # lat, lon, height,         geodesy
    ##                      # sza, saa, vza, vaa        geometry
    bbox = ((28,38), (-95,-75)) ## lat,lon preset for seus
    '''
    debug = True
    data_dir = Path("data")
    modis_swath_dir = data_dir.joinpath("modis")
    agg_dir = Path("/rstor/mdodson/aes770hw4/testing")

    year = 2019
    satellite = "terra"
    region = "seus"
    ## Max number of nearest modis to collect around each ceres footprint
    keep_closest = 400
    workers = 20
    #workers = 1

    """ One ceres pkl contains all of that year/satellite data by swath """
    ceres_swath_pkl = data_dir.joinpath(
            f"buffer/{satellite}_ceres_{region}_{year}.pkl")
    ceres_swaths = [FG1D(*s) for s in pkl.load(ceres_swath_pkl.open("rb"))]

    """ sort pkl paths to valid modis passes by increasing swath time """
    modis_swath_files = sorted([
            (parse_modis_dt(p),p)
            for p in data_dir.joinpath("modis").iterdir()
            if satellite in p.name
            if str(year) in p.name
            ], key=lambda t:t[0][0])

    #'''
    """
    Iterate over all available swaths, aggregate modis pixels, and store each
    swath as a new pkl object in agg_dir
    """
    with Pool(workers) as pool:
        swath_count = 0
        args = [{"swath_path":swath, "keep_closest":keep_closest,"debug":debug}
                for _,swath in modis_swath_files]
        for swath in pool.imap(mp_aggregate_ceres_modis, args):
            print(swath[0][1].shape, swath[1][1].shape)
            try:
                t0,tf = modis_swath_files[swath_count][0]
                stime = str(int((t0.timestamp() + t0.timestamp())/2))
                tmp_agg_pkl = agg_dir.joinpath(f"agg_{satellite}_{stime}.pkl")
                ## Write the (ceres, modis) swath
                pkl.dump(swath, tmp_agg_pkl.open("wb"))
            except Exception as e:
                #raise e
                print(e)
            finally:
                swath_count += 1
    #'''

    '''
    """ Single-file test run """
    ceres, modis = aggregate_ceres_modis(
            swath_path=modis_swath_files.pop(0)[1],
            keep_closest=300,
            debug=True,
            )
    print(ceres.shape, modis.shape)
    exit(0)
    '''

    '''
    """
    Collect and match overpass epoch times of each sensor collection to find
    missing data files. modis swath pkls have the ceres data already.
    """
    ceres_swath_times = np.array([
            np.average(C.data("epoch"))
            for C in ceres_swaths
            ])
    modis_swath_times = np.array([
            (t0.timestamp()+tf.timestamp())/2
            for t0,tf in [f[0] for f in modis_swath_files]
            ])
    modis_match_idx = [
            np.argmin(np.abs(ceres_swath_times-modtime))
            for modtime in modis_swath_times
            ]
    '''

    '''
    """
    Optionally check how close the average ceres to modis swath time is.
    Should be within a few seconds on average.
    """
    pair_times = [
            (modis_swath_times[i],ceres_swath_times[modis_match_idx[i]])
            for i in range(modis_swath_times.size)
            ]
    tdiffs = np.array([mt-ct for mt,ct in pair_times])
    '''

