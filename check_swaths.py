"""  """
import h5py
import random
import numpy as np
import json
import pickle as pkl
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from multiprocessing import Pool

def parse_swath_path(swath_h5:Path, time_as_epoch=False):
    """
    Each combined swath hdf5 file uniquely identifies its data with 3
    underscore-separated file name fields as follows:

    swath_{region}_{satellite}_{time}.h5

    this method parses these fields and returns them in order as a 3-tuple

    The first underscore-separated field doesn't have to be "swath", so the
    user can modify it to identify higher-level data categories; this method
    just ignores it entirely.
    """
    _,region,sat,date = swath_h5.stem.split("_")
    date = datetime.strptime(date, "%Y%m%d-%H%M")
    if time_as_epoch:
        date = int(date.timestamp())
    return region,sat,date

def check_swath(swath_h5:Path, image_dir:Path=None,
        get_tc:bool=False, get_dcp:bool=False, get_dust:bool=False):
    """
    Collect and return some useful bulk values and statistics on the CERES and
    MODIS data contained in a combined swath, and optionally generate some
    RGB images over the domain.

    per swath: region,satellite,datetime,shape
    per field: min max avg stdev nancount
    """
    region,sat,date = parse_swath_path(swath_h5)

    ## Load the swath with a buf_size_mb buffer
    f_swath = h5py.File(swath_h5, mode="r")
    modis_dict = json.loads(f_swath["data"].attrs["modis"])
    ceres_dict = json.loads(f_swath["data"].attrs["ceres"])
    modis = f_swath["/data/modis"][...]
    ceres = f_swath["/data/ceres"][...]
    f_swath.close()

    ## Save spatial shape of MODIS array
    mshape = modis.shape[:2]

    ## Flatten spatial dimensions of MODIS array
    modis = np.reshape(modis, (modis.shape[0]*modis.shape[1], modis.shape[2]))
    modis_stats = np.stack([
        np.nanmin(modis, axis=0),
        np.nanmax(modis, axis=0),
        np.nanmean(modis, axis=0),
        np.nanstd(modis, axis=0),
        np.count_nonzero(np.isnan(modis), axis=0).astype(float)
        ], axis=0)
    ceres_stats = np.stack([
        np.nanmin(ceres, axis=0),
        np.nanmax(ceres, axis=0),
        np.nanmean(ceres, axis=0),
        np.nanstd(ceres, axis=0),
        np.count_nonzero(np.isnan(ceres), axis=0).astype(float)
        ])
    swath_info = (region, sat, date, mshape)
    labels = (
            ## CERES string labels
            tuple(ceres_dict["labels"]),
            ## MODIS string labels
            tuple(modis_dict["flabels"]),
            ## Stats labels
            ("min","max","mean","stdev","nancount"),
            )
    return swath_info,ceres_stats,modis_stats,labels

def mp_check_swath(args):
    try:
        return check_swath(**args)
    except:
        print(f"FAILED: {args['swath_h5']}")

def check_multiple(ceres_swaths:list, output_pkl:Path=None, workers=1,
        image_dir=None, get_tc=False, get_dcp=False, get_dust=False):
    """ multiprocess collecting stats for all the swaths """
    mp_args = [{
        "swath_h5":s,
        "image_dir":image_dir,
        "get_tc":get_tc,
        "get_dcp":get_dcp,
        "get_dust":get_dust,
        } for s in ceres_swaths]
    swath_ids,ceres_stats,modis_stats = [],[],[]
    with Pool(workers) as pool:
        for result in pool.imap_unordered(mp_check_swath, mp_args):
            swath_tuple,cstats,mstats,labels = result
            swath_ids.append(swath_tuple)
            modis_stats.append(mstats)
            ceres_stats.append(cstats)
            print(f"Evaluated {swath_tuple}")
    ceres_stats = np.stack(ceres_stats, axis=0)
    modis_stats = np.stack(modis_stats, axis=0)
    out_tuple = (swath_ids,ceres_stats,modis_stats,labels)
    if not output_pkl is None:
        pkl.dump(out_tuple, output_pkl.open("wb"))
    return out_tuple

if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    ## directory where existingswaths files are
    #combined_swath_dir = data_dir.joinpath("swaths")
    combined_swath_dir = data_dir.joinpath("swaths")
    ## Path to a pickle file where swath-wise aggregate stats are placed
    out_pkl = data_dir.joinpath("swath-info_train.pkl")

    #rng = np.random.default_rng(seed=200007221752)
    rng = np.random.default_rng(seed=None)
    '''
    """ Dispatch a multiprocessed method to collect multiple swaths' data. """
    substrings = ("azn", "neus", "idn", "hkh", "seus", "alk",)
    swath_h5s = list(filter(
        lambda p:any(s in p.name for s in substrings),
        combined_swath_dir.iterdir()))
    check_multiple(
            ceres_swaths=swath_h5s,
            output_pkl=out_pkl,
            workers=23,
            )
    '''

    shuffle_swaths = True
    print_clabels = (
            "lat", "lon", "vza", "sza", "swflux", "lwflux",
            "pct_clr", "pct_l1", "pct_l2", "l1_cod", "l2_cod",
            "aer_land_pct", "aod_land",
            "aer_ocean_pct", "aod_ocean", "aod_ocean_small")
    print_mlabels = tuple(range(1,37))
    swaths,cstats,mstats,labels = pkl.load(out_pkl.open("rb"))
    clabels,mlabels,stat_labels = labels

    '''
    """ Print each of the swaths' bulk MODIS and CERES data """
    idx_swaths = sorted(list(enumerate(swaths)), key=lambda s:s[1][-2])
    if shuffle_swaths:
       rng.shuffle(idx_swaths)
    for i,s in idx_swaths:
        print()
        print(s)
        print(" "*16+"".join([f"{l:<16}" for l in stat_labels]))
        ## Print MODIS fields as rows of statistics
        for j,ml in enumerate(filter(lambda m:m in print_mlabels, mlabels)):
            pstr = f"{ml:<16}"
            pstr += "".join([f"{s:<16.3f}" for s in list(mstats[i,:,j])])
            print(pstr)
        ## Print CERES fields as rows of statistics
        for j,cl in enumerate(filter(lambda c:c in print_clabels, clabels)):
            pstr = f"{cl:<16}"
            pstr += "".join([f"{s:<16.3f}" for s in list(cstats[i,:,j])])
            print(pstr)
    '''

    agg_mstats = np.stack([
        np.amin(mstats[:,0], axis=0),
        np.amax(mstats[:,1], axis=0),
        np.average(mstats[:,2], axis=0),
        np.average(mstats[:,3], axis=0),
        np.round(np.sum(mstats[:,4], axis=0)).astype(int),
        ])
    agg_cstats = np.stack([
        np.amin(cstats[:,0], axis=0),
        np.amax(cstats[:,1], axis=0),
        np.average(cstats[:,2], axis=0),
        np.average(cstats[:,3], axis=0),
        np.round(np.sum(cstats[:,4], axis=0)).astype(int),
        ])
    print(f"## Bulk values calculated with {cstats.shape[0]} swaths")
    print("".join([f"{l:>16}," for l in ["field"]+list(stat_labels)]))
    ## Print MODIS fields as rows of statistics
    for j,ml in enumerate(filter(lambda m:m in print_mlabels, mlabels)):
        pstr = f"{ml:>16},"
        pstr += "".join([f"{s:>16.3f}," for s in list(agg_mstats[:,j])])
        print(pstr)
    ## Print CERES fields as rows of statistics
    for j,cl in enumerate(filter(lambda c:c in print_clabels, clabels)):
        pstr = f"{cl:>16},"
        pstr += "".join([f"{s:>16.3f}," for s in list(agg_cstats[:,j])])
        print(pstr)
