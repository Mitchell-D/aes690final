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
    pkl_path = Path("data/swath_stats.pkl")

    swath_stats = []
    ceres_labels = None

    #'''
    for p in swath_dir.iterdir():
        print(p.name)
        swath = h5py.File(p.open("rb"), "r")["data"]

        ceres_info = json.loads(swath.attrs["ceres"])
        ceres = FG1D(data=swath["ceres"][...], **ceres_info)

        if ceres_labels is None:
            ceres_labels = ceres.labels

        X = ceres.data()
        X[X>1.e20] = 0
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
    #'''

    #'''
    ceres_labels, satellites, regions, averages = pkl.load(pkl_path.open("rb"))
    print(averages.shape)
    #'''

    unique_regions = tuple(set(regions))
    unique_sats = tuple(set(satellites))

    ## Make a data cube with dimensions for (sat, region, datum)
    scube_shape = (len(unique_sats), len(unique_regions), len(ceres_labels))
    scube = np.full(scube_shape, 0.)

    sat_idxs = tuple(unique_sats.index(s) for s in satellites)
    reg_idxs = tuple(unique_regions.index(r) for r in regions)
    for i,(sidx,ridx) in enumerate(zip(sat_idxs, reg_idxs)):
        scube[sidx,ridx] += averages[i]
    scube /= averages.shape[0]

    for i,l in enumerate(ceres_labels):
        print(l, np.average(averages[...,i]))

    '''
    rstats = {r:{s:{l:[] for l in ceres_labels} for s in ("aqua", "terra")}
              for r in unique_regions}
    isterra = np.array(map(lambda s:s=="terra", satellites))
    for i,(s,r) in enumerate(zip(satellites,regions)):
        for j,l in enumerate(ceres_labels):
            rstats[r][s][l].append(averages[i,j])
    '''
