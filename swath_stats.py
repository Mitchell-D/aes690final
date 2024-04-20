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
        X = np.average(X, axis=0)

        swath_stats.append((ceres.meta["satellite"], ceres.meta["region"], X))

    satellite,region,averages = zip(*swath_stats)
    averages = np.stack(averages, axis=0)
    pkl.dump((ceres_labels, satellite, region, averages), pkl_path.open("wb"))
    #'''

    '''
    ceres_labels, satellite, region, averages = pkl.load(pkl_path.open("rb"))
    print(averages.shape)
    print(len(ceres_labels), len(satellite), len(region))
    '''

