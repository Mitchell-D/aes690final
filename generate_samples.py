"""
Module containing data generators that load data from swath hdf5 files created
by get_modis_swath, extract MODIS subgrids around each CERES footprint,
and calculate the CERES PSF over the MODIS subgrids.
"""
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

import geom_utils as gu
from FG1D import FG1D
from FeatureGridV2 import FeatureGridV2 as FG

def _psf_analytic(xi):
    """
    Analytic PSF function from CERES ATBD subsystem 4.4 eq 2 with respect to
    the xi angle calculated in terms of beta/delta and the FOV boundaries.

    xi is expected to be provided in radians (!!) >:O
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
    ## Sometimes values are anomalously high; the cross-track scan angle should
    ## never be more than a few degrees, so just clip it to avoid overflow.
    ## The PSF will evaluate to zero at this distance in any case.
    abs_beta = np.clip(np.abs(beta), 0., 6.)
    delta = np.clip(delta, -6., 6.)
    #print(np.amin(abs_beta), np.amax(abs_beta), np.amin(delta), np.amax(delta))
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
    psf_sum = np.sum(np.sum(psf,axis=1,keepdims=True),axis=2,keepdims=True)
    return psf/psf_sum

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
    #return np.stack((psf, beta, delta), axis=-1)
    return psf

def ndsnap(points, latlon):
    """
    Snap an 2D-array of points to values along an 2D-array grid.
    Each point will be snapped to the grid value with the smallest
    city-block distance.

    :@param points: (P,2) array of P (lat,lon) points
    :@param grid: (M,N,2) array of M latitude and N longitude values
        represented by features (lat, lon)

    :@return: A 2D-array with one row per row of points. Each i-th row will
        correspond to row of grid to which the i-th row of points is closest.
        In case of ties, it will be snapped to the row of grid with the
        smaller index.
    """
    grid = np.reshape(latlon, (-1,2))
    grid_3d = np.transpose(grid[:,:,np.newaxis], [2,1,0])
    diffs = np.sum(np.abs(grid_3d - points[:,:,np.newaxis]), axis=1)
    best = np.argmin(diffs, axis=1)
    return divmod(best, latlon.shape[1])

def swaths_dataset(
        swath_h5s:list,
        modis_feats:list=None,
        ceres_labels:tuple=("swflux","lwflux"),
        ceres_feats:tuple=("sza", "vza", "raa"),
        modis_feats_norm:tuple=None,
        ceres_labels_norm:tuple=None,
        ceres_feats_norm:tuple=None,
        modis_grid_size=48,
        num_swath_procs=1,
        samples_per_swath=128,
        block_size=32,
        buf_size_mb=128,
        deterministic=False,
        mask_val=None,
        seed=None,
        debug=False,
        **kwargs,
        ):
    """
    Opens multiple combined swath hdf5s (made by get_modis_swath) as
    dataset generators, and interleaves their results.

    :@param swath_h5s: Paths to combined swath hdf5 files to generate the data
    :@param modis_grid_size: Side length of the generated MODIS tile domain
    :@param modis_feats: MODIS band labels used for features, in order.
    :@param modis_feats_norm: 2-tuple (offset,gain) of None or array with
        size=len(modis_feats) values to normalize each MODIS band like
        (band-offset)/gain.
    :@param ceres_labels_norm: CERES flux norm bounds as in modis_feats_norm
    :@param ceres_feats_norm: CERES geometry norm bounds as in modis_feats_norm
    :@param ceres_labels: CERES bands that are predicted.
    :@param ceres_feats: CERES geometry bands that are used for features.
    :@param num_swath_procs: Number of swath generators to multithread over
    :@param samples_per_swath: Maximum number of footprints to yield from a
        single swath. If swaths have fewer than this number that's okay.
    :@param block_size: Number of consecutive samples drawn per swath
    :@param buf_size_mb: hdf5 buffer size for each swath file in MB
    :@param deterministic: If True, always yields block_size samples per swath
        at a time; if False, multiple swaths may inconsistently interleave.
    :@param seed: Seed
    """
    if modis_feats == None:
        modis_feats = list(range(1,37))

    ## output like ((modis, geometry, psf), ceres)
    ## (B,N,N,Fm) MODIS observation
    modis_shape = (modis_grid_size, modis_grid_size, len(modis_feats))
    ## (B,N,N,Fg) geometry from CERES
    geom_shape = (modis_grid_size, modis_grid_size, len(ceres_feats),)
    ## (B,N,N,1)) point spread function
    psf_shape = (modis_grid_size,modis_grid_size,1)
    ## (B,Fc) CERES observation
    ceres_shape = (len(ceres_labels),)
    out_sig = ((
        tf.TensorSpec(shape=modis_shape, dtype=tf.float64),
        tf.TensorSpec(shape=geom_shape, dtype=tf.float64),
        tf.TensorSpec(shape=psf_shape, dtype=tf.float64),
        ),tf.TensorSpec(shape=ceres_shape, dtype=tf.float64))

    def _gen_swath(swath_path):
        """
        Generator for a single swath file

        1. Open the swath file and initialize FeatureGrid objects of the data
        2. Randomly select up to samples_per_swath CERES footprints
        3. Determine the lat/lon of the closest pixel to the CERES centroid,
           then extract a modis_grid_size square around it. Drop any samples
           where the MODIS grid extends out of bounds.
        4. Extract MODIS radiances, CERES fluxes, and CERES geometry associated
           with each square sub-domain in the sample selection.
        5. Calculate the point spread function over the domain.
        """

        """ loading all the data from the provided swath file """

        ## Determines the buffer size by assuming each chunk isn't much bigger
        ## than 1MB. There are probably better ways to tune this.
        swath_path = swath_path.decode()
        f_swath = h5py.File(
                swath_path,
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
        #print(clatlon.shape, mlatlon.shape)
        #print(np.average(clatlon, axis=0), np.average(np.average(mlatlon, axis=0), axis=0))
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
        G = ceres.data(ceres_feats)[idxs]
        C = ceres.data(ceres_labels)[idxs]
        ## Make slices corresponding to each MODIS tile and extract the bands
        tile_slices = [
                (slice(lb_latlon[i,0], ub_latlon[i,0]),
                 slice(lb_latlon[i,1], ub_latlon[i,1]))
                for i in range(lb_latlon.shape[0])]
        M = modis.data(modis_feats)
        M = np.stack([M[lb,ub] for lb,ub in tile_slices], axis=0)

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
                [modis_latlon[lb,ub] for lb,ub in tile_slices],
                axis=0)
        P = calc_psf(
                ceres_latlon=np.delete(ceres_latlon, np.where(oob), axis=0),
                modis_latlon=modis_latlon,
                subsat_latlon=np.delete(subsat_latlon, np.where(oob), axis=0),
                )
        G = np.broadcast_to(G[:,np.newaxis,np.newaxis,:],
                            (G.shape[0], *geom_shape))
        P = P[...,np.newaxis]

        print(f"{P.shape[0]} samples taken from {swath_path}")

        """ Normalize per feature if requested """
        if not modis_feats_norm is None:
            moff,mgain = map(np.asarray, modis_feats_norm)
            M = (M-moff)/mgain
        if not ceres_feats_norm is None:
            goff,ggain = map(np.asarray, ceres_feats_norm)
            G = (G-goff)/ggain
        if not ceres_labels_norm is None:
            coff,cgain = map(np.asarray, ceres_labels_norm)
            C = (C-coff)/cgain

        ## check for NaNs
        if not mask_val is None:
            nans = None
            if np.any(nans:=np.logical_not(np.isfinite(M))):
                if debug:
                    print(f"Replacing {np.count_nonzero(nans)} MODIS "
                          f"non-finite values found in {swath_path}")
                M[nans] = mask_val
            if np.any(nans:=np.logical_not(np.isfinite(C))):
                if debug:
                    print(f"Replacing {np.count_nonzero(nans)} CERES "
                          f"non-finite values found in {swath_path}")
                C[nans] = mask_val
            if np.any(nans:=np.logical_not(np.isfinite(G))):
                if debug:
                    print(f"Replacing {np.count_nonzero(nans)} Geometry "
                          f"non-finite values found in {swath_path}")
                G[nans] = mask_val
            if np.any(nans:=np.logical_not(np.isfinite(P))):
                if debug:
                    print(f"Replacing {np.count_nonzero(nans)} PSF "
                          f"non-finite values found in {swath_path}")
                P[nans] = mask_val

        """ yield results """
        for i in range(lb_latlon.shape[0]):
            x = tuple(map(tf.convert_to_tensor, (M[i], G[i], P[i])))
            y = tf.convert_to_tensor(C[i])
            yield (x,y)

    ## Convert the swath paths to open in each generator
    swath_h5s = tf.data.Dataset.from_tensor_slices(
            list(map(lambda p:p.as_posix(), map(Path,swath_h5s))))

    """
    Concurrently open num_swath_procs swath hdf5s as generator datasets,
    and cycle through the swaths, building a dataset by consuming
    block_size elements at once from each. When a generator is depleted,
    open a new swath in its place until swath_h5s is depleted.

    For more details see:
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#for_example
    """
    D = swath_h5s.interleave(
            lambda fpath: tf.data.Dataset.from_generator(
                generator=_gen_swath,
                args=(fpath,),
                output_signature=out_sig,
                ),
            cycle_length=num_swath_procs,
            num_parallel_calls=num_swath_procs,
            block_length=block_size,
            deterministic=deterministic,
            )
    return D

def get_tiles_h5(
        tile_h5_path:Path, swath_h5s, modis_feats, ceres_feats, ceres_labels,
        modis_grid_size, samples_per_swath=1000000, block_size=1000000,
        modis_feats_norm:tuple=None, ceres_feats_norm:tuple=None,
        ceres_labels_norm:tuple=None, num_swath_procs=1, mask_val=None,
        deterministic=False, batch_chunk_size=256, max_tile_count=None,
        buf_size_mb=128, debug=False, seed=None):
    """
    """
    tile_h5_path = Path(tile_h5_path)
    if tile_h5_path.exists():
        print(f"Warning: {tile_h5_path.as_posix()} exits!")
        return tile_h5_path
    with h5py.File(tile_h5_path, "w") as f:
        chunk_shape = (batch_chunk_size, modis_grid_size, modis_grid_size)
        DP = f.create_dataset(
                name="/data/psf",
                shape=(0, modis_grid_size, modis_grid_size, 1),
                maxshape=(None, modis_grid_size, modis_grid_size, 1),
                chunks=(*chunk_shape, 1)
                )
        DG = f.create_dataset(
                name="/data/geom",
                shape=(0,modis_grid_size,modis_grid_size,len(ceres_feats)),
                maxshape=(None,modis_grid_size,
                          modis_grid_size,len(ceres_feats)),
                chunks=(*chunk_shape, len(ceres_feats)),
                compression="gzip",
                )
        DM = f.create_dataset(
                name="/data/modis",
                shape=(0, modis_grid_size, modis_grid_size, len(modis_feats)),
                maxshape=(None, modis_grid_size,
                          modis_grid_size, len(modis_feats)),
                chunks=(*chunk_shape, len(modis_feats)),
                compression="gzip",
                )
        DC = f.create_dataset(
                name="/data/ceres",
                shape=(0, len(ceres_labels)),
                maxshape=(None, len(ceres_labels)),
                chunks=(batch_chunk_size, len(ceres_labels)),
                )
        swath_dataset_params = {
                "swath_h5s":list(map(lambda p:p.as_posix(), swath_h5s)),
                "modis_feats":modis_feats,
                "ceres_feats":ceres_feats,
                "ceres_labels":ceres_labels,
                "modis_grid_size":modis_grid_size,
                "num_swath_procs":num_swath_procs,
                "samples_per_swath":samples_per_swath,
                "block_size":block_size,
                "modis_feats_norm":tuple(map(tuple,modis_feats_norm)),
                "ceres_labels_norm":tuple(map(tuple,ceres_labels_norm)),
                "ceres_feats_norm":tuple(map(tuple,ceres_feats_norm)),
                "mask_val":mask_val,
                "deterministic":deterministic,
                "buf_size_mb":buf_size_mb,
                "seed":seed,
                }
        f["data"].attrs.update({
            "swath_dataset_params":json.dumps(swath_dataset_params)
            })
        dataset = swaths_dataset(debug=debug, **swath_dataset_params)
        h5idx = 0
        for (m,g,p),c in dataset.batch(batch_chunk_size):
            s = slice(h5idx, h5idx+m.shape[0])
            h5idx += m.shape[0]
            DM.resize((h5idx, *DM.shape[1:]))
            DG.resize((h5idx, *DG.shape[1:]))
            DP.resize((h5idx, *DP.shape[1:]))
            DC.resize((h5idx, *DC.shape[1:]))
            DM[s,...] = m.numpy()
            DG[s,...] = g.numpy()
            DP[s,...] = p.numpy()
            DC[s,...] = c[...].numpy()
            f.flush()
        f.close()
    return tile_h5_path

def tiles_dataset(
        tiles_h5s:list, modis_feats, ceres_feats, ceres_labels,
        buf_size_mb=128., num_tiles_procs=1, block_size=1, deterministic=True,
        **kwargs):
    """
    Returns a tensorflow dataset that interleaves data from one or more
    tiles files that were previously created by get_tiles_h5

    BEWARE this method doesn't check whether the normalization coefficients
    are consistent between the tiles h5s.
    """

    if isinstance(tiles_h5s, str):
        tiles_h5s = Path(tiles_h5s)
    if isinstance(tiles_h5s,Path):
        tiles_h5s = [tiles_h5s]

    ## Load the attribute dictionaries containing each tiles h5's parameters
    ## which were used to call swath_dataset for the tiles h5 creation.
    tiles_attrs = {}
    tiles_h5s = list(map(Path, tiles_h5s))
    for p in tiles_h5s:
        f = h5py.File(p, mode="r")
        tmp_attrs = json.loads(f["data"].attrs["swath_dataset_params"])
        tiles_attrs[p.as_posix()] = tmp_attrs
        f.close()
    sample_attrs = list(tiles_attrs.values())[0]

    ## infer the output shapes from the first tiles attrs dict.
    grid_shape = (sample_attrs["modis_grid_size"],
                  sample_attrs["modis_grid_size"])
    out_sig = ((
        tf.TensorSpec(shape=(*grid_shape, len(modis_feats)), dtype=tf.float64),
        tf.TensorSpec(shape=(*grid_shape, len(ceres_feats)), dtype=tf.float64),
        tf.TensorSpec(shape=(*grid_shape, 1), dtype=tf.float64),
        ),tf.TensorSpec(shape=(len(ceres_labels),), dtype=tf.float64))

    def _gen_tiles(tiles_file):
        """
        yields ((modis, geom, psf), ceres) tensors from a single tile file
        """
        tiles_file = tiles_file.decode()
        f_swath = h5py.File(
                tiles_file,
                mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15,
                )
        info = json.loads(f_swath["data"].attrs["swath_dataset_params"])
        ## make sure all the requested bands are present by checking the
        ## parameters used to initialize swaths_dataset
        assert all(l in info["modis_feats"] for l in modis_feats),tiles_file
        assert all(l in info["ceres_feats"] for l in ceres_feats),tiles_file
        assert all(l in info["ceres_labels"] for l in ceres_labels),tiles_file

        midx = tuple(info["modis_feats"].index(l) for l in modis_feats)
        gidx = tuple(info["ceres_feats"].index(l) for l in ceres_feats)
        cidx = tuple(info["ceres_labels"].index(l) for l in ceres_labels)

        ## Load the hdf5 datasets
        DM = f_swath["/data/modis"]
        DG = f_swath["/data/geom"]
        DP = f_swath["/data/psf"]
        DC = f_swath["/data/ceres"]

        ## extract one chunk at a time along batch dimension and yield data
        ## with the features ordered as requested.
        for c in DM.iter_chunks():
            tmp_m = tf.convert_to_tensor(DM[c[0],...][...,midx])
            tmp_g = tf.convert_to_tensor(DG[c[0],...][...,gidx])
            tmp_p = tf.convert_to_tensor(DP[c[0],...])
            tmp_c = tf.convert_to_tensor(DC[c[0],...][...,cidx])
            for i in range(tmp_m.shape[0]):
                yield ((tmp_m[i], tmp_g[i], tmp_p[i]), tmp_c[i])
        f_swath.close()


    tiles_h5s = tf.data.Dataset.from_tensor_slices(
            list(map(lambda p:p.as_posix(), map(Path,tiles_h5s))))
    D = tiles_h5s.interleave(
            lambda fpath: tf.data.Dataset.from_generator(
                generator=_gen_tiles,
                args=(fpath,),
                output_signature=out_sig,
                ),
            cycle_length=num_tiles_procs,
            num_parallel_calls=num_tiles_procs,
            block_length=block_size,
            deterministic=deterministic,
            )
    return D

def get_modis_mosaic(m):
    """

    """
    mside = int(np.ceil(m.shape[-1]**(1/2)))
    missing = mside**2 - m.shape[-1]
    m = np.concatenate((m, np.zeros((*m.shape[:3],missing))), axis=-1)
    mosaics = []
    for i in range(m.shape[0]):
        mosaics.append(np.concatenate([
            np.concatenate([
                gaussnorm(m[i,...,j*mside+k]) for k in range(mside)
                ], axis=1)
            for j in range(mside)
            ], axis=0))
    return np.stack(mosaics, axis=0)

if __name__=="__main__":
    """
    The code below consists of a few basic implementations of each method in
    this module which are mainly for generating tiles hdf5 files, and for
    debugging / sanity checks.
    """
    debug = True
    data_dir = Path("data")
    fig_dir = Path("figures")

    swath_dir = data_dir.joinpath("swaths")
    #swath_dir = data_dir.joinpath("swaths_val")


    """
    Unless you're generating images, you can probably comment these imports.
    """
    #from plot_swath import gaussnorm
    #from krttdkit.operate import enhance as enh
    #from krttdkit.visualize import guitools as gt
    #from krttdkit.visualize import geoplot as gp

    """ don't comment these imports """
    from norm_coeffs import modis_norm,ceres_norm,geom_norm
    (mlabels,mnorm),(clabels,cnorm),(glabels,gnorm) = map(
            lambda t:zip(*t), (modis_norm, ceres_norm, geom_norm))


    #'''
    """ Generate a new tiles hdf5 file from swath hdf5s """
    #tiles_dir = data_dir.joinpath("tiles_train")
    tiles_dir = data_dir.joinpath("tiles_val")
    swath_substrings = ("aqua","seus")
    swath_name = "-".join(swath_substrings)+"-val"
    tiles_h5_path = tiles_dir.joinpath(f"tiles_{swath_name}.h5")

    swath_h5s = list(filter(
        lambda p:all(s in p.name for s in swath_substrings),
        swath_dir.iterdir(),
        ))
    get_tiles_h5(
            tile_h5_path=tiles_h5_path,
            swath_h5s=swath_h5s,
            modis_feats=mlabels,
            ceres_feats=glabels,
            ceres_labels=clabels,
            modis_grid_size=48,
            samples_per_swath=256,
            block_size=8, ## deplete the swath per block
            modis_feats_norm=tuple(map(np.array, zip(*mnorm))),
            ceres_feats_norm=tuple(map(np.array, zip(*gnorm))),
            ceres_labels_norm=tuple(map(np.array, zip(*cnorm))),
            num_swath_procs=4,
            mask_val=-9999.,
            deterministic=True,
            batch_chunk_size=128,
            max_tile_count=None,
            buf_size_mb=128,
            seed=None,
            debug=debug,
            )
    exit(0)
    #'''

    #'''
    """ Load a tiles dataset from an existing tiles file """
    tiles_h5s = [
            data_dir.joinpath(f"tiles_aqua_test-val.h5"),
            data_dir.joinpath(f"tiles_aqua_test-train.h5"),
            #data_dir.joinpath(f"tiles_terra_test-val.h5"),
            #data_dir.joinpath(f"tiles_terra_test-train.h5"),
            ]
    tds = tiles_dataset(
            tiles_h5s=tiles_h5s,
            #modis_feats=(8,1,4,3,2,18,26,7,20,28,30,31,33,24,25),
            modis_feats=mlabels,
            ceres_feats=("sza", "vza"),
            ceres_labels=("swflux", "lwflux"),
            buf_size_mb=128.,
            num_tiles_procs=5,
            block_size=4,
            deterministic=False,
            )

    '''
    """ Use the tiles dataset to make a RGB mosaic of MODIS values"""
    mosaic_path = fig_dir.joinpath("modis/mosaics")
    mosaic_count = 0
    mosaic_limit = 64
    rgb_norm = lambda X:np.floor(
        np.clip((X-np.amin(X))/np.ptp(X),0,.9999)*256
        ).astype(np.uint8)
    for (m,g,p),c in tds.batch(256):
        m = m.numpy()
        mosaics = get_modis_mosaic(m)
        for i in range(mosaics.shape[0]):
            #gt.quick_render(gt.scal_to_rgb(mosaics[i]))
            mosaic_count += 1
            if mosaic_count > mosaic_limit:
                #exit(0)
                break
            gp.generate_raw_image(
                    rgb_norm(gt.scal_to_rgb(mosaics[i])),
                    mosaic_path.joinpath(f"mosaic_aqua_{mosaic_count:02}.png")
                    )
    exit(0)
    '''

    '''
    """ Generate data in real time from swath hdf5s  """
    g = swaths_dataset(
            swath_h5s=[s for s in swath_dir.iterdir()],
            buf_size_mb=512,
            modis_grid_size=48,
            num_swath_procs=3,
            samples_per_swath=16,
            block_size=2,
            modis_feats=None,
            ceres_labels=("swflux","lwflux"),
            ceres_feats=("sza", "vza"),
            **kwargs
            )

    bidx = 0
    for ((m,g,p),c) in g.prefetch(2).batch(16):
        #for i in range(p.shape[0]):
        #    print(enh.array_stat(p[i,...,0].numpy()))
        #    gt.quick_render(gt.scal_to_rgb(p[i,...,0]))
        for j in range(m.shape[0]):
            cstr = "-".join([
                f"{v:03}" for v in tuple(np.array(c[j]).astype(np.uint16))
                ])
            X = np.clip(m[j,:,:,:3], 0, 1)*255
            X = X.astype(np.uint8)
            #gt.quick_render(X, vmax=256)
            exit(0)
            gp.generate_raw_image(
                    np.array(X),
                    fig_dir.joinpath(
                        f"modis_tile/{bidx:02}-{j:03}_{cstr}.png"),
                    )
        bidx += 1
    '''
