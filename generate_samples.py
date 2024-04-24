"""  """
#import gc
import sys
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

'''
def _delta_f(beta):
    b = np.abs(beta)
    if b<a:
        return -a
    elif b<2*a:
        return -2*a + b
    else:
        raise ValueError("Condition should never be met; God help you")
'''

def _psf_analytic(xi):
    """
    Analytic PSF function from CERES ATBD subsystem 4.4 eq 2
    """
    ## analytic psf constants
    c1 = 1.98412
    a1 = 1.84205
    a2 = -0.22502

    term_1 = 1 - (1 + a1 + a2) * np.exp(-1 * c1 * xi)

    b1 = 1.47034
    t2exp = -6.35465
    t2ang = 1.90282
    term_2 = a1 * np.cos(t2ang * xi)
    term_2 += b1 * np.sin(t2ang * xi)
    term_2 *= np.exp(t2exp * xi)

    b2 = 0.45904
    t3exp = -4.61598
    t3ang = 5.83072
    term_3 = a2 * np.cos(t3ang * xi)
    term_3 += b2 * np.sin(t3ang * xi)
    term_3 *= np.exp(t3exp * xi)

    return term_1 + term_2 + term_3

def _psf_conditionals(beta, delta):
    """
    Conditional PSF function from ATBD subsystem 4.4 eq 1
    """
    a = .65 ## FOV angular bound
    ## PSF is symmetric over cross-track scan line
    abs_beta = np.abs(beta)
    ## Describes the boundary of the hexagonal optical FOV
    d_f = np.where(abs_beta<a, -1*a, -2*a+abs_beta)
    ## PSF within the optical hexagonal boundary
    psf_in_fov = _psf_analytic(delta - d_f)
    ## Blurred PSF leading the optical hexagon
    psf_before_fov = psf_in_fov - _psf_analytic(delta + d_f)
    psf = np.where(delta<-1*d_f, psf_in_fov, psf_before_fov)
    ## Clip values where cross-scan angle is too wide
    psf[abs_beta>2*a] = 0#np.amin(psf)
    ## Clip values after the along-scan boundary
    psf[delta<d_f] = 0#np.amin(psf)
    ## Clip values below zero
    psf[psf<0] = 0
    return psf/np.sum(psf, axis=0, keepdims=True)

def calc_psf(ceres_latlon, modis_latlon, subsat_latlon,
         sensor_altitude=705., earth_radius=6367.):
    """
    Calculate the CERES point spread function over a series of CERES centroids
    associated with MODIS grid tiles, described by their lat/lon values and
    those of the subsatellite (nadir) point.

    :@param ceres_latlon: (B, 2) array of (lat,lon) points at B centroids
    :@param modis_latlon: (B, L, L, 2) array of (lat,lon) points at B tiles
        each consisting of a LxL grid of pixels which should coincide with
        the corresponding CERES centroids.
    :@param subsat_latlon: (B, 2) array of (lat,lon) points at B sub-satellite
        instances corresponding to each of the CERES/MODIS observations.

    :@return: (B, L, L, 3) array of (PSF, beta, delta) values associated with
        the point spread function and angles calculated over the MODIS domain.
    """
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

    ## Calculate the PSF over the domain
    ## (!!!) TODO: (!!!)
    ## need to change delta based on Z' sign, which should be opposite scan dir
    psf = _psf_conditionals(beta, delta + .96)
    return np.stack((psf, beta, delta), axis=-1)

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
    """ """
    if modis_bands == None:
        modis_bands = list(range(1,37))

    ## output like ((modis, geometry, psf), ceres)
    modis_shape = (modis_grid_size, modis_grid_size, len(modis_bands))
    geom_shape = (len(ceres_geom),)
    psf_shape = (modis_grid_size,modis_grid_size,3)
    ceres_shape = (len(ceres_pred),)
    out_sig = ((
        tf.TensorSpec(shape=modis_shape, dtype=tf.float64),
        tf.TensorSpec(shape=geom_shape, dtype=tf.float64),
        tf.TensorSpec(shape=psf_shape, dtype=tf.float64),
        ), tf.TensorSpec(shape=ceres_shape, dtype=tf.float64))

    def _gen_swath(swath_path):
        """ Generator for a single swath file """

        """ loading all the data from the provided swath file """

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
        modis = FG(data=f_swath["/data/modis"][...], **modis_dict)
        ceres = FG1D(data=f_swath["/data/ceres"][...], **ceres_dict)
        f_swath.close()

        """ choosing CERES centroids for this swath """

        ## Randomly extract a number of samples from the swath and snap them to
        ## the closest point in the MODIS grid in order to describe a subgrid.
        rng = np.random.default_rng(seed=seed)
        idxs = np.arange(ceres.size)
        rng.shuffle(idxs)
        idxs = idxs[:samples_per_swath]
        clatlon = ceres.data(("lat", "lon"))[idxs]
        mlatlon = modis.data(("lat", "lon"))
        cen_latlon = ndsnap(clatlon, mlatlon)

        """ MODIS tile boundary identification """

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

        """ input data extraction """

        ## Extract the viewing geometry and ceres footprints
        G = ceres.data(ceres_geom)[idxs]
        C = ceres.data(ceres_pred)[idxs]
        ## Make slices corresponding to each MODIS tile and extract the bands
        tile_slices = [
                (slice(lb_latlon[i,0], ub_latlon[i,0]),
                 slice(lb_latlon[i,1], ub_latlon[i,1]))
                for i in range(lb_latlon.shape[0])]
        M = modis.data(modis_bands)
        M = np.stack([M[*ts] for ts in tile_slices], axis=0)

        """ point spread function calculation """

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
                [modis_latlon[*ts] for ts in tile_slices],
                axis=0)
        PSF = calc_psf(
                ceres_latlon=ceres_latlon,
                modis_latlon=modis_latlon,
                subsat_latlon=subsat_latlon,
                )

        """ yielding results """

        print(G.shape, C.shape, M.shape, PSF.shape)

        for i in range(lb_latlon.shape[0]):
            X = tuple(map(tf.convert_to_tensor, (M[i], G[i], PSF[i])))
            Y = tf.convert_to_tensor(C[i])
            yield (X, Y)
            '''
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
            '''

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
    modis_swath_dir = data_dir.joinpath("swaths")

    g = gen_swaths_samples(
            swath_h5s=[s for s in modis_swath_dir.iterdir()],
            buf_size_mb=512,
            modis_grid_size=48,
            num_swath_procs=3,
            samples_per_swath=16,
            block_size=2,
            modis_bands=(26,1,6),
            #modis_bands=None,
            ceres_pred=("swflux","lwflux"),
            ceres_geom=("sza", "vza", "raa"),
            )
    bidx = 0
    for ((m,g,p),c) in g.prefetch(2).batch(16):
        '''
        #for i in range(p.shape[0]):
            print(enh.array_stat(p[i,...,0].numpy()))
            gt.quick_render(gt.scal_to_rgb(p[i,...,0]))
        '''
        for j in range(m.shape[0]):
            cstr = "-".join([
                f"{v:03}" for v in tuple(np.array(c[j]).astype(np.uint16))
                ])
            X = np.clip(m[j,:,:,:3], 0, 1)*255
            X = X.astype(np.uint8)
            #gt.quick_render(X, vmax=256)
            gp.generate_raw_image(
                    np.array(X),
                    fig_dir.joinpath(
                        f"modis_tile/{bidx:02}-{j:03}_{cstr}.png"),
                    )
        bidx += 1
