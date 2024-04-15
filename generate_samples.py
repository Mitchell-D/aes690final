"""  """
#import gc
import random
import numpy as np
import h5py
import tensorflow as tf
import json

from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from collections import ChainMap

from krttdkit.operate import enhance as enh
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

from FG1D import FG1D
from FeatureGridV2 import FeatureGridV2 as FG

'''
def get_modis_swath(ceres_swath:FG1D, laads_token:str, modis_nc_dir:Path,
                    swath_h5_dir:Path, region_label:str,
                    lat_buffer:float=0., lon_buffer:float=0.,
                    bands:tuple=None, isaqua=False, keep_rad=False,
                    keep_masks=False, spatial_chunks:tuple=(64,64),
                    debug=False):
'''

def ndsnap(points, latlon):
    """
    Adapted from:
    https://stackoverflow.com/questions/8457645/efficient-pythonic-way-to-snap-a-value-to-some-grid
    Snap an 2D-array of points to values along an 2D-array grid.
    Each point will be snapped to the grid value with the smallest
    city-block distance.

    Parameters
    ---------
    points: (P,2) array of P (lat,lon) points
    grid: (M,N,2) array of M latitude and N longitude values represented by
        features (lat, lon)

    Returns
    -------
    A 2D-array with one row per row of points. Each i-th row will
    correspond to row of grid to which the i-th row of points is closest.
    In case of ties, it will be snapped to the row of grid with the
    smaller index.
    """
    grid = np.reshape(latlon, (-1,2))
    grid_3d = np.transpose(grid[:,:,np.newaxis], [2,1,0])
    diffs = np.sum(np.abs(grid_3d - points[:,:,np.newaxis]), axis=1)
    best = np.argmin(diffs, axis=1)
    return divmod(best, latlon.shape[1])

def gen_swaths_samples(
        swath_h5s:list,
        buf_size_mb=128,
        modis_grid_size=48,
        modis_bands:list=(8,1,4,3,2,18,5,26,6,7,20,27,28,30,31,33),
        ceres_pred:tuple=("swflux","lwflux"),
        ceres_geom:tuple=("sza", "vza", "raa"),
        num_swath_procs=1,
        samples_per_swath=128,
        block_size=32, ## Number of consecutive samples drawn per swath
        seed=None
        ):
    """
    """
    #h5_path = swath_h5_dir.joinpath(
    #        f"swath_{region_label}_{sat}_{timestr}.h5")
    if modis_bands == None:
        modis_bands = list(range(1,37))

    out_sig = ((
        tf.TensorSpec(
            shape=(modis_grid_size,modis_grid_size,len(modis_bands)),
            dtype=tf.float64,
            ),
        tf.TensorSpec(shape=(len(ceres_geom),), dtype=tf.float64),
        ), tf.TensorSpec(shape=(len(ceres_pred),), dtype=tf.float64))

    def _gen_swath(swath_path):
        """ Generator for a single swath file """
        ## Determines the buffer size by assuming each chunk isn't much bigger
        ## than 1MB. There are probably better ways to tune this.
        f_swath = h5py.File(
                swath_path.decode(),
                mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15,
                )
        modis_dict = json.loads(f_swath["data"].attrs["modis"])
        ceres_dict = json.loads(f_swath["data"].attrs["ceres"])

        ## Add new batch coordinate for (B,Q,Q,M) shaped MODIS grid batches
        #modis_dict.update(clabels=("b", *modis_dict.get("clabels")))

        modis = FG(
                data=f_swath["/data/modis"][...],
                **modis_dict
                )
        ceres = FG1D(data=f_swath["/data/ceres"][...], **ceres_dict)

        f_swath.close()

        ## Randomly extract samples from the swath and snap them to
        rng = np.random.default_rng(seed=seed)
        idxs = np.arange(ceres.size)
        rng.shuffle(idxs)
        idxs = idxs[:samples_per_swath]
        clatlon = ceres.data(("lat", "lon"))[idxs]
        mlatlon = modis.data(("lat", "lon"))
        cen_latlon = ndsnap(clatlon, mlatlon)

        ## Extract a modis_grid_size square around each centroid and make sure
        ## the indeces are in bounds
        lb_latlon = np.transpose(np.array(cen_latlon) - int(modis_grid_size/2))
        ub_latlon = lb_latlon + modis_grid_size
        oob = np.any(np.logical_or(
            lb_latlon<0, ub_latlon>np.array(mlatlon.shape[:2])
            ), axis=-1)
        if np.any(oob):
            print(f"oob: {np.where(oob)}")
        lb_latlon = np.delete(lb_latlon, np.where(oob), axis=0)
        ub_latlon = np.delete(ub_latlon, np.where(oob), axis=0)

        ## Extract the geometry, ceres footprints, and full modis grid
        G = ceres.data(ceres_geom)[idxs]
        C = ceres.data(ceres_pred)[idxs]
        M = modis.data(modis_bands)

        print(swath_path.decode(), lb_latlon.shape[0])
        for i in range(lb_latlon.shape[0]):
            ## Subset the modis grid for the current tile
            m = M[lb_latlon[i,0]:ub_latlon[i,0],lb_latlon[i,1]:ub_latlon[i,1]]
            yield ((tf.convert_to_tensor(m), tf.convert_to_tensor(G[i])),
                   tf.convert_to_tensor(C[i]))

    ## Establish a dataset of swath paths to open in each generator
    swath_h5s = tf.data.Dataset.from_tensor_slices(
            list(map(lambda p:p.as_posix(), swath_h5s)),
            )
    ## Open num_swath_procs hdf5s at a time and interleave their results,
    ## consuming block_size training samples at each stage.
    D = swath_h5s.interleave(
            lambda fpath: tf.data.Dataset.from_generator(
                generator=_gen_swath,
                args=(fpath,),
                output_signature=out_sig,
                ),
            cycle_length=num_swath_procs,
            num_parallel_calls=num_swath_procs,
            block_length=block_size,
            )
    return D

if __name__=="__main__":
    debug = False
    data_dir = Path("data")
    fig_dir = Path("figures")
    modis_swath_dir = data_dir.joinpath("modis_swaths")

    g = gen_swaths_samples(
            swath_h5s=[s for s in modis_swath_dir.iterdir()],
            buf_size_mb=512,
            modis_grid_size=96,
            num_swath_procs=8,
            samples_per_swath=4,
            block_size=2,
            modis_bands=(1,4,3,"lat","lon","sza","vza"),
            ceres_pred=("swflux","lwflux"),
            ceres_geom=("sza", "vza", "raa","lat","lon"),
            )
    bidx = 0
    for ((m,g),c) in g.prefetch(2).batch(64):
        for j in range(c.shape[0]):
            print(tuple(np.array(c[j]).astype(np.uint16)))
            cstr = "-".join([
                f"{v:03}" for v in tuple(np.array(c[j]).astype(np.uint16))
                ])
            #print(np.cos(np.deg2rad(m[j,:,:,-2])))
            #brdf = np.cos(np.deg2rad(m[j,:,:,-2]))
            #brdf = np.stack([brdf for i in range(3)], axis=-1)
            X = np.clip(m[j,:,:,:3], 0, 1)*255
            X = X.astype(np.uint8)
            #gt.quick_render(X, vmax=256)
            gp.generate_raw_image(
                    np.array(X),
                    fig_dir.joinpath(
                        f"modis_tile/{bidx:02}-{j:03}_{cstr}.png"),
                    )
        bidx += 1
