import numpy as np
import pickle as pkl
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

from krttdkit.operate import enhance as enh
from krttdkit.visualize import guitools as gt

if __name__=="__main__":
    swaths_pkl = Path("data/modis_swaths/swath_idn_aqua_20200108-0528.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_neus_aqua_20180113-1832.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_neus_aqua_20181215-1831.h5")
    #swaths_pkl = Path("data/modis_swaths/swath_hkh_aqua_20181203-0813.h5")
    swath = h5py.File(swaths_pkl.open("rb"), "r")["data"]
    ceres = swath["ceres"]
    modis = swath["modis"]

    gt.quick_render(np.dstack([
        modis[...,0],
        modis[...,4],
        modis[...,3],
        ]))

    print(modis)
    print(ceres)
