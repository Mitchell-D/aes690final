import numpy as np
import pickle as pkl
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

from krttdkit.operate import enhance as enh
from krttdkit.visualize import guitools as gt

from FeatureGridV2 import FeatureGridV2
from FG1D import FG1D

def feature_stats(modis:FeatureGridV2, feature_labels=None):
    if feature_labels is None:
        feature_labels = modis.flabels
    for fl in feature_labels:
        metrics = ("mean", "stddev", "min", "max",
                   "nanmin", "nanmax", "nancount")
        tmp_stat = enh.array_stat(modis.data(fl))
        tmp_str = ", ".join([f"{m}:{tmp_stat[m]:.4f}" for m in metrics])
        print(fl,tmp_str)

if __name__=="__main__":
    swaths_pkl = Path("data/combined_swaths/swath_neus_terra_20201214-1551.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_neus_aqua_20180113-1832.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_neus_aqua_20181215-1831.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_hkh_aqua_20181203-0813.h5")
    swath = h5py.File(swaths_pkl.open("rb"), "r")["data"]

    modis = FeatureGridV2(
            data=swath["modis"][...],
            **json.loads(swath.attrs["modis"])
            )
    ceres = FG1D(
            data=swath["ceres"][...],
            **json.loads(swath.attrs["ceres"])
            )

    print(modis.flabels)
    print(f"Center wavelengths:", modis.meta("ctr_wls"))
    print(ceres.labels)
    print(ceres.meta.keys())

    '''
    gt.quick_render(np.dstack([
        modis.data(1),
        modis.data(4),
        modis.data(3),
        ]))
    '''

    '''
    gt.quick_render(np.dstack([
        modis.data(26)**1/.66,
        modis.data(1),
        modis.data(6),
        ]))
    '''

    ## pixel equatorial coordinates
    #rgb = modis.data(("x_img","y_img","z_img"))

    ## truecolor
    #rgb = modis.data((1,4,3))

    ## day cloud phase
    rgb = modis.data((26,1,6))

    rgb = (rgb-np.amin(rgb))/np.ptp(rgb)
    gt.quick_render(rgb)

    feature_stats(modis, (1,35,26))

    print(modis)
    print(ceres)
