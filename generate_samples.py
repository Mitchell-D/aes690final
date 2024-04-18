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

import geom_utils as gu
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

## These lists define the feature ordering anticipated by the PSF function
psf_modis_labels = ['x_img', 'y_img', 'z_img']
psf_ceres_labels = ['x_sat', 'y_sat', 'z_sat',
                    'x_cen', 'y_cen', 'z_cen',
                    'xx_s2c', 'xy_s2c', 'xz_s2c',
                    'yx_s2c', 'yy_s2c', 'yz_s2c',
                    'zx_s2c', 'zy_s2c', 'zz_s2c']

def psf(psf_ceres, modis_equatorial):
    """
    """

    print(psf_ceres.shape, modis_equatorial.shape)
    ## Satellite equatorial vectors (B, 1, 1, 9)
    sat_eq = psf_ceres[...,:3]
    ## CERES satellite-relative viewing vectors (B, 1, 1, 9)
    ceres_vv = psf_ceres[..., 6:]
    modis_grid_shape = modis_equatorial.shape[:-1]
    mod_eq = modis_equatorial.reshape((-1,modis_equatorial.shape[-1]))
    '''
    print(np.average(np.average(modis_equatorial, axis=1),axis=1)[0])
    print(sat_eq[0])
    print()
    print(np.average(np.average(modis_equatorial, axis=1),axis=1)[1])
    print(sat_eq[1])
    print()
    print(np.average(np.average(modis_equatorial, axis=1),axis=1)[2])
    print(sat_eq[2])
    print()
    '''

    ## MODIS satellite-relative viewing vectors (B, L, L, 9)
    modis_vv = gu.get_view_vectors(sat_eq, modis_equatorial)
    #'''
    print(ceres_vv.shape, modis_vv.shape)
    print(np.average(np.average(modis_vv[0], axis=0), axis=0))
    print(ceres_vv[0])
    print()
    print(np.average(np.average(modis_vv[3], axis=0), axis=0))
    print(ceres_vv[3])
    print()
    print(np.average(np.average(modis_vv[6], axis=0), axis=0))
    print(ceres_vv[6])
    #'''
    exit(0)

def psf2(ceres_latlon, modis_latlon, subsat_latlon,
         sensor_altitude=705., earth_radius=6367.):
    ## Get equatorial vectors for the satellite, centroid, and imager pixel
    eq_sat = gu.get_equatorial_vectors( ## shape: (B, 3)
            latitude=subsat_latlon[...,0],
            longitude=subsat_latlon[...,1],
            )
    eq_cer = gu.get_equatorial_vectors( ## shape: (B, 3)
            latitude=ceres_latlon[...,0],
            longitude=ceres_latlon[...,1],
            )
    eq_mod = gu.get_equatorial_vectors( ## shape: (B, L, L, 3)
            latitude=modis_latlon[...,0],
            longitude=modis_latlon[...,1],
            )
    ## Returns orthogonal equatorial (X', Y', Z'), each with (x,y,z) components
    ## shape: 3 of (B,1,1,3)
    CX,CY,CZ = np.split(np.expand_dims(gu.get_view_vectors(
            sensor_equatorial_vectors=eq_sat,
            pixel_equatorial_vectors=eq_cer,
            sensor_altitude=sensor_altitude,
            earth_radius=earth_radius,
            ), axis=(1,2)), 3, axis=-1)
    MX,MY,MZ= np.split(gu.get_view_vectors( ## shape: 3 of (B,L,L,3)
            sensor_equatorial_vectors=np.expand_dims(eq_sat, axis=(1,2)),
            pixel_equatorial_vectors=eq_mod,
            sensor_altitude=sensor_altitude,
            earth_radius=earth_radius,
            ), 3, axis=-1)

    ## Calculate along (delta) and across (beta) track angles
    delta = np.rad2deg(np.arcsin(np.sum(MY*CZ, axis=-1)))
    tmp = np.cross(CZ, MY)
    tmp /= np.linalg.norm(tmp, axis=-1, keepdims=True)
    beta = np.rad2deg(np.arcsin(-1.*np.sum(tmp*CY, axis=-1)))

    a = .65 ## angular bounds
    ## analytic psf constants
    a1 = 1.84205
    a2 = -0.22502
    b1 = 1.47034
    b2 = 0.45904
    c1 = 1.98412
    t2exp = -6.35465
    t2ang = 1.90282
    t3exp = -4.61598
    t3ang = 5.83072

    def _delta_f(beta):
        b = np.abs(beta)
        if b<a:
            return -a
        elif b<2*a:
            return -2*a + b
        else:
            raise ValueError("Condition should never be met; God help you")
    def _delta_b(beta):
        b = np.abs(beta)
        if b<a:
            return a
        elif b<2*a:
            return 2*a - b
        else:
            raise ValueError("Condition should never be met; God help you")

    def _psf_analytic(xi):
        """
        Analytic PSF function from CERES ATBD subsystem 4.4 eq 2
        """
        term_1 = 1 - (1 + a1 + a2) * np.exp(-1 * c1 * xi)

        term_2 = a1 * np.cos(np.deg2rad(t2ang * xi))
        term_2 += b1 * np.sin(np.deg2rad(t2ang * xi))
        term_2 *= np.exp(t2exp * xi)

        term_3 = a2 * np.cos(np.deg2rad(t3ang * xi))
        term_3 += b2 * np.sin(np.deg2rad(t3ang * xi))
        term_3 *= np.exp(t3exp * xi)

        return term_1 + term_2 + term_3

    def _psf_conditionals(beta, delta):
        """
        Conditional PSF function from ATBD subsystem 4.4 eq 1
        """
        if np.abs(beta)>2*a or delta<(d_f:=_delta_f(beta)):
            return 0

        elif

        psf = np.zeros_like(beta)


    for i in range(delta.shape[0]):
        print(enh.array_stat(delta[i]))
        print(enh.array_stat(beta[i]))
        print()
    exit(0)

    '''
    modis_geom = gu.get_sensor_pixel_geometry(
            nadir_lat=subsat_latlon[...,0],
            nadir_lon=subsat_latlon[...,1],
            obsv_lat=modis_latlon[...,0],
            obsv_lon=modis_latlon[...,1],
            )
    ceres_geom = gu.get_sensor_pixel_geometry(
            nadir_lat=subsat_latlon[...,0],
            nadir_lon=subsat_latlon[...,1],
            obsv_lat=ceres_latlon[...,0],
            obsv_lon=ceres_latlon[...,1],
            )
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
        modis_bands:list=None,
        #modis_bands:list=(8,1,4,3,2,18,5,26,6,7,20,27,28,30,31,33),
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

        ## Randomly extract a number of samples from the swath and snap them to
        ## the closest point in the MODIS grid in order to describe a subgrid.
        rng = np.random.default_rng(seed=seed)
        idxs = np.arange(ceres.size)
        rng.shuffle(idxs)
        idxs = idxs[:samples_per_swath]
        clatlon = ceres.data(("lat", "lon"))[idxs]
        mlatlon = modis.data(("lat", "lon"))
        cen_latlon = ndsnap(clatlon, mlatlon)

        ## Extract a modis_grid_size square around each centroid
        ## and make sure the indeces are in bounds.
        lb_latlon = np.transpose(
                np.array(cen_latlon).astype(int) - int(modis_grid_size/2))
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

        PSF_GEOM_C = ceres.data(psf_ceres_labels)[idxs]
        tile_slices = [
                (slice(lb_latlon[i,0], ub_latlon[i,0]),
                 slice(lb_latlon[i,1], ub_latlon[i,1]))
                for i in range(lb_latlon.shape[0])]
        M = modis.data(modis_bands)
        M = np.stack([M[*ts] for ts in tile_slices], axis=0)
        PSF_GEOM_M = modis.data(psf_modis_labels)
        PSF_GEOM_M = np.stack([PSF_GEOM_M[*ts] for ts in tile_slices], axis=0)

        ## Convert CERES to longitude in [0,360)
        ceres_latlon = np.stack(
                (ceres.data("lat")[idxs],
                 (ceres.data("lon")[idxs]+360.)%360.),
                axis=-1
                )
        ## Convert CERES subsatellite colatitude to latitude in [-90,90]
        ## subsatellite longitude is already in [0,360)
        subsat_latlon = np.stack(
                (90-ceres.data("ndr_colat")[idxs],
                 ceres.data("ndr_lon")[idxs]),
                axis=-1)
        ## Convert MODIS to longitude in [0,360)
        modis_latlon = np.stack(
                (modis.data("lat"),
                 (modis.data("lon")+360.)%360.),
                axis=-1)
        modis_latlon = np.stack(
                [modis_latlon[*ts] for ts in tile_slices],axis=0)

        PSF = psf2(
                ceres_latlon=ceres_latlon,
                modis_latlon=modis_latlon,
                subsat_latlon=subsat_latlon,
                )
        exit(0)

        for i in range(lb_latlon.shape[0]):
            #psf(PSF_GEOM_C[i], PSF_GEOM_M[i])
            exit(0)
            ## Subset the modis grid for the current tile
            m = M[lb_latlon[i,0]:ub_latlon[i,0],
                  lb_latlon[i,1]:ub_latlon[i,1]]
            psf_m = PSF_GEOM_M[lb_latlon[i,0]:ub_latlon[i,0],
                               lb_latlon[i,1]:ub_latlon[i,1]]
            ## return tensors like ((modis, geometry), ceres)
            print(swath_path.decode())

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
    modis_swath_dir = data_dir.joinpath("swaths_test")

    g = gen_swaths_samples(
            swath_h5s=[s for s in modis_swath_dir.iterdir()],
            buf_size_mb=512,
            modis_grid_size=32,
            num_swath_procs=1,
            samples_per_swath=32,
            block_size=8,
            modis_bands=None,
            ceres_pred=("swflux","lwflux"),
            ceres_geom=("sza", "vza", "raa"),
            )
    bidx = 0
    for ((m,g),c) in g.prefetch(2).batch(16):
        for j in range(c.shape[0]):
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
