import numpy as np
import pickle as pkl
import h5py
import json
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint as ppt

from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

from plot_ceres import geo_scatter
from FeatureGridV2 import FeatureGridV2
from FG1D import FG1D

def gaussnorm(X, contrast=None, gamma=1):
    """
    Gauss-normalize the array, clip it at the `contrast` number of standard
    deviations, then normalize it to [0,1]. Then apply the gamma correction.
    """
    if (r:=np.std(X)) == 0.:
        X = np.zeros(X.shape)
    else:
        X = (X-np.average(X))/r
        if not contrast is None:
            X = np.clip(X, -contrast, contrast)
        X = (X-np.amin(X))/np.ptp(X)
        X = X**(1/gamma)
    return X

def rgb_norm(X):
    return np.floor(
            np.clip((X-np.amin(X))/np.ptp(X),0,.9999)*256
            ).astype(np.uint8)

if __name__=="__main__":
    swath_dir = Path("data/swaths_val")
    fig_dir = Path("figures/swaths")

    rng = np.random.default_rng(seed=None)
    swath_paths = list(swath_dir.iterdir())
    rng.shuffle(swath_paths)

    swath_paths = [ Path("data/swaths/swath_neus_aqua_20200530-1807.h5") ]

    #'''
    """ Load MODIS and CERES FeatureGridV2 and FG1D objects and print info """
    while len(swath_paths) > 0:
        swath = h5py.File(swath_paths.pop(0).open("rb"), "r")["data"]
        modis_info = json.loads(swath.attrs["modis"])
        ceres_info = json.loads(swath.attrs["ceres"])
        print(modis_info.keys())
        modis = FeatureGridV2(data=swath["modis"][...], **modis_info)
        ceres = FG1D(data=swath["ceres"][...], **ceres_info)
        print(ceres.labels)
        print(modis.flabels)

        timestr = datetime.fromtimestamp(
                int(np.average(ceres.data("epoch")))
                ).strftime("%Y%m%d-%H%M")
        lat_range = ceres.meta["lat_range"]
        lon_range = ceres.meta["lon_range"]
        regionstr = ceres.meta["region"]
        satstr = ceres.meta["satellite"]
        #'''

        #'''
        """ Plot the CERES data corresponding to the domain """
        geo_scatter(
            ceres_fg1d=ceres,
            clabel="swflux",
            show=True,
            #fig_path=fig_dir.joinpath(f"geo_scatter_{timestr}_swflux.png"),
            fig_path=fig_dir.joinpath(
                f"{regionstr}_{timestr}_{satstr}_flux-sw.png"),
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
            fig_path=fig_dir.joinpath(
                f"{regionstr}_{timestr}_{satstr}_flux-lw.png"),
            plot_spec={
                "title":f"Longwave full-sky flux (W/m^2) {timestr}",
                "marker_size":200,
                "marker":",",
                "text_size":16,
                "cbar_shrink":.6,
                },
            )
        #'''

        #'''
        """ Make a bool mask of MODIS pixels outside the CERES bounds """
        m_lat = np.logical_or(
                modis.data("lat")<lat_range[0],
                modis.data("lat")>lat_range[1]
                )
        m_lon = np.logical_or(
                modis.data("lon")<lon_range[0],
                modis.data("lon")>lon_range[1]
                )
        m_oob = np.logical_or(m_lat, m_lon)
        #'''

        #'''
        """ Make day cloud phase, truecolor, and dust RGBs """
        ## Select image contrast and gamma parameters
        contrast = 7 ## clip values further than this many standard deviations
        gamma_dcp,gamma_tc,gamma_dust = 2,3,.4
        '''
        rgb_norm = lambda X:np.floor(
                np.clip((X-np.amin(X))/np.ptp(X),0,.9999)*256
                ).astype(np.uint8)
        '''
        rgb_dcp = rgb_norm(np.dstack(list(map(
            lambda X:gaussnorm(X, contrast=contrast, gamma=gamma_dcp),
            [modis.data(26)**1/.66,modis.data(1),modis.data(6)]
            ))))
        rgb_tc = rgb_norm(np.dstack(list(map(
            lambda X:gaussnorm(X, contrast=contrast, gamma=gamma_tc),
            [modis.data(1),modis.data(4),modis.data(3)]
            ))))
        rgb_dust = rgb_norm(np.dstack(list(map(
            lambda X:gaussnorm(X, contrast=contrast, gamma=gamma_dust),
            [modis.data(32)-modis.data(31),
             modis.data(31)-modis.data(29),
             modis.data(31)]
            ))))
        ## Mask out-of-bounds
        rgb_dcp[m_oob] = 0
        rgb_tc[m_oob] = 0
        rgb_dust[m_oob] = 0

        #'''
        ## show the user (comment if no cv2)
        gt.quick_render(rgb_dcp)
        gt.quick_render(rgb_tc)
        gt.quick_render(rgb_dust)
        #'''

        #'''
        ## generate figures
        gp.generate_raw_image(
                rgb_dcp,
                fig_dir.joinpath(
                    f"{regionstr}_{timestr}_{satstr}_rgb-dcp.png")
                )
        gp.generate_raw_image(
                rgb_tc,
                fig_dir.joinpath(
                    f"{regionstr}_{timestr}_{satstr}_rgb-tc.png")
                )
        gp.generate_raw_image(
                rgb_dust,
                fig_dir.joinpath(
                    f"{regionstr}_{timestr}_{satstr}_rgb-dust.png")
                )
        #'''
