"""
Script for extracting basic statistics from CERES swath pkl files,
and for printing out useful information about the CERES swaths.
"""
import numpy as np
import pickle as pkl
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from FG1D import FG1D

if __name__=="__main__":
    swath_dir = Path("data/swaths")
    pkl_path = Path("data/ceres_stats.pkl")

    '''
    ## Extract bulk statistics region-wise
    swath_stats = []
    for p in swath_dir.iterdir():
        print(p.name)
        swath = h5py.File(p.open("rb"), "r")["data"]

        ceres_info = json.loads(swath.attrs["ceres"])
        ceres = FG1D(data=swath["ceres"][...], **ceres_info)

        ceres_labels = ceres.labels
        X = ceres.data()
        X[X>1.e20] = 0 ## Ignore masked values (ie where no COD/AOD)

        X = np.asarray([
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            np.average(X, axis=0),
            np.std(X, axis=0),
            ])
        swath_stats.append((ceres.meta["satellite"], ceres.meta["region"], X))

    satellites,regions,stats = zip(*swath_stats)
    stats = np.stack(stats, axis=0)
    pkl.dump((ceres_labels, satellites, regions, stats), pkl_path.open("wb"))
    exit(0)
    '''

    """ Load the labels, sat/region strings, and (N,4,F) stats arra y"""
    ceres_labels, satellites, regions, stats = pkl.load(pkl_path.open("rb"))
    print(stats.shape)
    unique_regions = tuple(set(regions))
    unique_sats = tuple(set(satellites))

    '''
    """ Print the number of valid swaths per region """
    region_counts = [(y,np.count_nonzero(np.array([x==y for x in regions])))
                     for y in unique_regions]
    for r,c in region_counts:
        print(r,c)
    '''

    """ Make a data cube with dimensions for (sat, region, stat, datum) """
    scube_shape = (len(unique_sats), len(unique_regions),
                   stats.shape[1], len(ceres_labels))
    scube = np.full(scube_shape, 0.)

    sat_idxs = tuple(unique_sats.index(s) for s in satellites)
    reg_idxs = tuple(unique_regions.index(r) for r in regions)
    counts = np.zeros((*scube_shape[:2], 1, 1)) ## counts for averaging
    for i,(sidx,ridx) in enumerate(zip(sat_idxs, reg_idxs)):
        scube[sidx,ridx,:,:] += stats[i]
        counts[sidx,ridx] += 1
    scube /= counts

    #'''
    """ Print the relevant bulk statistics region-wise """
    print_labels = ("lat", "lon", "vza", "sza", "raa", "swflux", "lwflux",
                    "pct_clr", "pct_l1", "pct_l2", "l1_cod", "l2_cod",
                    "aer_land_pct", "aod_land", "aer_ocean_pct", "aod_ocean",
                    "aod_ocean_small")
    stat_columns = ("Label", "Min", "Max", "Avg", "Std")
    for ridx,r in enumerate(unique_regions):
        print(f"\n{r}")
        print("".join(f"{v:<16}" for v in stat_columns))
        for l in print_labels:
            lidx = ceres_labels.index(l)
            printstr = "{:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(
                    *list(np.average(scube[:,ridx,:,lidx], axis=0)))
            print(f"{l:<15} {printstr}")
    #'''
