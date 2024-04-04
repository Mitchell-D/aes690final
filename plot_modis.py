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
from krttdkit.visualize import geoplot as gp

from plot_ceres import geo_scatter
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

def gaussnorm(X,contrast=None):
    X = (X-np.average(X))/np.std(X)
    if contrast is None:
        return X
    return np.clip(X, -contrast, contrast)

if __name__=="__main__":
    #swaths_pkl = Path("data/combined_swaths/swath_neus_aqua_20180801-1744.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_hkh_aqua_20180105-0748.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_neus_aqua_20190426-1807.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_idn_aqua_20190224-0516.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_idn_aqua_20191118-0458.h5")

    #swaths_pkl = Path("data/combined_swaths/swath_idn_aqua_20191118-0458.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_idn_aqua_20200401-0504.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_azn_aqua_20200229-1814.h5")
    swaths_pkl = Path("data/combined_swaths/swath_azn_aqua_20180311-1814.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_hkh_aqua_20180525-0813.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_hkh_aqua_20200514-0813.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_idn_aqua_20191118-0458.h5")
    #swaths_pkl = Path("data/combined_swaths/swath_neus_terra_20201128-1551.h5")
    fig_dir = Path("figures/swaths")

    swath = h5py.File(swaths_pkl.open("rb"), "r")["data"]

    modis = FeatureGridV2(
            data=swath["modis"][...],
            **json.loads(swath.attrs["modis"])
            )
    ceres = FG1D(
            data=swath["ceres"][...],
            **json.loads(swath.attrs["ceres"])
            )

    #'''
    timestr = datetime.fromtimestamp(
            int(np.average(ceres.data("epoch")))
            ).strftime("%Y-%m-%dT%H%M")
    lat_range = ceres.meta["lat_range"]
    lon_range = ceres.meta["lon_range"]
    regionstr = ceres.meta["region"]

    geo_scatter(
        ceres_fg1d=ceres,
        clabel="swflux",
        show=True,
        #fig_path=fig_dir.joinpath(f"geo_scatter_{timestr}_swflux.png"),
        fig_path=fig_dir.joinpath(f"{regionstr}_{timestr}_flux-sw.png"),
        plot_spec={
            "title":f"Shortwave full-sky flux (W/m^2) {timestr}",
            "marker_size":200,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            }
        )
    geo_scatter(
        ceres_fg1d=ceres,
        clabel="lwflux",
        show=True,
        #fig_path=fig_dir.joinpath(f"geo_scatter_{timestr}_lwflux.png"),
        fig_path=fig_dir.joinpath(f"{regionstr}_{timestr}_flux-lw.png"),
        plot_spec={
            "title":f"Longwave full-sky flux (W/m^2) {timestr}",
            "marker_size":200,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            },
        )
    #'''

    print(modis.flabels)
    print(ceres.labels)

    m_lat = np.logical_or(
            modis.data("lat")<lat_range[0],
            modis.data("lat")>lat_range[1]
            )
    m_lon = np.logical_or(
            modis.data("lon")<lon_range[0],
            modis.data("lon")>lon_range[1]
            )
    m_oob = np.logical_or(m_lat, m_lon)

    '''
    gt.quick_render(np.dstack([
        modis.data(26)**1/.66,
        modis.data(1),
        modis.data(6),
        ]))
    '''

    ## pixel equatorial coordinates
    #rgb = modis.data(("x_img","y_img","z_img"))

    #'''
    contrast=6
    ## day cloud phase
    gamma = 1
    rgb = np.dstack([gaussnorm(modis.data(b)**(1/gamma),contrast=contrast)
                     for b in (26,1,6)])
    rgb = ((rgb-np.amin(rgb))/np.ptp(rgb)*255).astype(np.uint8)
    rgb[m_oob] = 0
    gt.quick_render(rgb)
    gp.generate_raw_image(
            rgb, fig_dir.joinpath(f"{regionstr}_{timestr}_rgb-dcp.png"))
    #'''

    #'''
    ## truecolor
    gamma = 4
    rgb = np.dstack([gaussnorm(modis.data(b)**(1/gamma), contrast=contrast)
                     for b in (1,4,3)])
    rgb = ((rgb-np.amin(rgb))/np.ptp(rgb)*255).astype(np.uint8)
    rgb[m_oob] = 0
    gt.quick_render(rgb)
    gp.generate_raw_image(
            rgb, fig_dir.joinpath(f"{regionstr}_{timestr}_rgb-tc.png"))
    #'''

    #'''
    gamma = .4
    rgb = np.stack([
            gaussnorm((modis.data(32)-modis.data(31)),
                      contrast=contrast),
            gaussnorm((modis.data(31)-modis.data(29)),
                      contrast=contrast),
            gaussnorm(modis.data(31), contrast=contrast),
            ],axis=-1)
    rgb = (rgb-np.amin(rgb))/np.ptp(rgb)
    rgb = (rgb**(1/gamma)*255).astype(np.uint8)
    rgb[m_oob] = 0
    gt.quick_render(rgb)
    gp.generate_raw_image(
            rgb, fig_dir.joinpath(f"{regionstr}_{timestr}_rgb-dust.png"))
    #'''
