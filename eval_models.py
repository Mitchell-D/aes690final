"""
Script for randomly searching a user-defined combinatorial graph
of model configurations within the tracktrain framework.
"""

import numpy as np
import json
import tensorflow as tf
import h5py
from pathlib import Path
from pprint import pprint as ppt
from datetime import datetime
import tensorflow as tf
import pickle as pkl
import gc

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train


from generate_samples import swaths_dataset
from check_swaths import parse_swath_path
from norm_coeffs import modis_norm,ceres_norm,geom_norm
modis_norm,geom_norm,ceres_norm = map(dict,(modis_norm,geom_norm,ceres_norm))
from FeatureGridV2 import FeatureGridV2
from FG1D import FG1D
from plot_ceres import geo_scatter
from plot_swath import gaussnorm,rgb_norm

from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

"""
Config contains configuration values default to all models.
This config may be added to and some fields may be overwritten downstream.
"""

def eval_model(model_dir:ModelDir):
    pass

def load_swath(swath_path:Path):
    swath = h5py.File(swath_path.open("rb"), "r")["data"]
    modis_info = json.loads(swath.attrs["modis"])
    ceres_info = json.loads(swath.attrs["ceres"])
    modis = FeatureGridV2(data=swath["modis"][...], **modis_info)
    ceres = FG1D(data=swath["ceres"][...], **ceres_info)
    return ceres,modis

def load_model(model_dir:Path, weights_path:Path=None):
    """
    Load a PAED or CEDA style model from its weights
    """
    md = ModelDir(Path(model_dir))
    model = md.load_weights(weights_path)
    grid_layer_names = ("join_dec-agg",)
    all_layer_names = [l.name for l in model.layers]
    grid_output = next(
            model.get_layer(s).output for s in grid_layer_names
            if s in all_layer_names
            )
    model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.output, grid_output]
            )
    return md,model

def get_swaths_dataset(
        swaths_h5s:list, model_config:dict, samples_per_swath=128,
        block_size=16, num_procs=1
        ):
    """
    Get a tensorflow dataset generating swaths from the provided list, given
    the configuration defined as a json in a corresponding model dict
    (or any custom-defined parameters from generate_samples.swaths_dataset)
    """
    model_config["samples_per_swath"] = samples_per_swath
    model_config["block_size"] = block_size
    model_config["num_swath_procs"] = num_procs
    return swaths_dataset(swath_h5s=swaths_h5s, **model_config)

def eval_model_on_full_swath(md, model, ceres, modis):
    """ """
    mmean,mstd = zip(*[modis_norm[k] for k in md.config["modis_feats"]])
    gmean,gstd = zip(*[geom_norm[k] for k in md.config["ceres_feats"]])
    cmean,cstd = zip(*[ceres_norm[k] for k in md.config["ceres_labels"]])

    m = modis.data(md.config["modis_feats"])[np.newaxis,...]
    m = (m-mmean)/mstd
    ## trick to process large array row-wise by swapping with the batch axis.
    m = tf.transpose(m,(1,0,2,3))
    #c = ceres.data(md.config["ceres_labels"])
    #g = np.average(
    #        ceres.data(md.config["ceres_feats"]), axis=0
    #        )[np.newaxis,np.newaxis,np.newaxis,:]
    #g = np.broadcast_to(g, (*m.shape[:-1], g.shape[-1]))
    #my,mx = m.shape[1:3]
    ## use average geometry conditions for rough estimate since MODIS
    ## geometry scale is currently messed up.
    g = np.zeros((*m.shape[:-1],2))
    g = np.zeros((*m.shape[:-1],len(md.config["ceres_feats"])))
    # Fully express every pixel
    p = np.full((*m.shape[:-1],1), 1, dtype=float)
    p /= float(p.shape[1]*p.shape[2])

    batch_size = 256
    agg_disagg = []
    for i in range(0, m.shape[0], batch_size):
        with tf.device('/cpu:0'):
            slc = slice(i,min((i+batch_size, m.shape[0])))
            agg_disagg.append(model((m[slc],g[slc],p[slc])))
    agg,disagg = zip(*agg_disagg)
    agg = tf.concat(agg, axis=0)
    disagg = tf.concat(disagg, axis=0)
    agg = agg*np.array(cstd)+np.array(cmean)
    disagg = disagg*np.array(cstd)+np.array(cmean)
    return np.average(agg, axis=0),tf.transpose(disagg,(1,0,2,3))

def eval_swath_grid(swath_path:Path, model_dir:Path, fig_dir:Path=None):
    ## Load the model
    md,model = load_model(model_dir)
    ## Load the swath
    ceres,modis = load_swath(sp)

    mmean,mstd = zip(*[modis_norm[k] for k in md.config["modis_feats"]])
    gmean,gstd = zip(*[geom_norm[k] for k in md.config["ceres_feats"]])
    cmean,cstd = zip(*[ceres_norm[k] for k in md.config["ceres_labels"]])

    sdata = get_swaths_dataset(
            swaths_h5s=[sp],
            model_config=md.config,
            samples_per_swath=10000,
            block_size=16,
            num_procs=1,
            )
    C,D = [],[]
    for (m,g,p),c in sdata.batch(32):
        with tf.device('/cpu:0'):
            agg,disagg = model((m,g,p))
        D.append(agg.numpy())
        C.append(c.numpy())
    del sdata

    C = np.concatenate(C, axis=0)*np.array(cstd)+np.array(cmean)
    D = np.concatenate(D, axis=0)*np.array(cstd)+np.array(cmean)

    timestr = datetime.fromtimestamp(
            int(np.average(ceres.data("epoch")))
            ).strftime("%Y%m%d %H%Mz")
    lat_range = ceres.meta["lat_range"]
    lon_range = ceres.meta["lon_range"]

    ## evaluate on the full grid
    agg,disagg = eval_model_on_full_swath(md, model, ceres, modis)

    del model

    print(agg.shape, disagg.shape)

    if fig_dir is None:
        return agg,disagg

    ## Make a bool mask of MODIS pixels outside the CERES bounds
    modis_latlon = np.stack((modis.data("lat"),modis.data("lon")),axis=-1)
    oob = modis
    m_lat = np.logical_or(
            modis.data("lat")<lat_range[0],
            modis.data("lat")>lat_range[1]
            )
    m_lon = np.logical_or(
            modis.data("lon")<lon_range[0],
            modis.data("lon")>lon_range[1]
            )
    m_not_oob = np.logical_not(np.logical_or(m_lat, m_lon))

    ## generate some RBGs
    contrast = 7
    gamma_dcp,gamma_tc,gamma_dust = 2,3,.4
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

    #rgb_dcp[np.logical_not(m_not_oob)] = 0
    #rgb_tc[np.logical_not(m_not_oob)] = 0
    #rgb_dust[np.logical_not(m_not_oob)] = 0
    gp.generate_raw_image(rgb_tc, fig_dir.joinpath(sp.stem+"_rgb-tc.png"))
    gp.generate_raw_image(rgb_dcp, fig_dir.joinpath(sp.stem+"_rgb-dcp.png"))
    gp.generate_raw_image(rgb_dust, fig_dir.joinpath(sp.stem+"_rgb-dust.png"))
    del modis, rgb_tc, rgb_dcp, rgb_dust
    gc.collect()

    da_1d = np.concatenate(
            (np.squeeze(disagg)[m_not_oob], modis_latlon[m_not_oob]),
            axis=-1
            )
    da_fg1d = FG1D(labels=("swflux", "lwflux", "lat", "lon"), data=da_1d)

    geo_scatter(
        ceres_fg1d=da_fg1d,
        clabel="swflux",
        show=False,
        fig_path=fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_da-sw.png"),
        plot_spec={
            "title":f"Disagg SW flux (W/m^2) {md.dir.name} {timestr}",
            "marker_size":5,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            }
        )
    geo_scatter(
        ceres_fg1d=da_fg1d,
        clabel="lwflux",
        show=False,
        fig_path=fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_da-lw.png"),
        plot_spec={
            "title":f"Disagg LW flux (W/m^2) {md.dir.name} {timestr}",
            "marker_size":5,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            },
        )
    del da_fg1d
    geo_scatter(
        ceres_fg1d=ceres,
        clabel="swflux",
        show=False,
        fig_path=fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_obs-sw.png"),
        plot_spec={
            "title":f"CERES shortwave full-sky flux (W/m^2) {timestr}",
            "marker_size":280,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            }
        )
    geo_scatter(
        ceres_fg1d=ceres,
        clabel="lwflux",
        show=False,
        fig_path=fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_obs-lw.png"),
        plot_spec={
            "title":f"CERES longwave full-sky flux (W/m^2) {timestr}",
            "marker_size":280,
            "marker":",",
            "text_size":16,
            "cbar_shrink":.6,
            },
        )
    del ceres
    gc.collect()

    swflux_rgb = rgb_norm(gt.scal_to_rgb(np.squeeze(disagg)[...,0]))
    lwflux_rgb = rgb_norm(gt.scal_to_rgb(np.squeeze(disagg)[...,1]))
    gp.generate_raw_image(
            swflux_rgb,
            fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_rgb-sw.png"))
    del swflux_rgb
    gp.generate_raw_image(
            lwflux_rgb,
            fig_dir.joinpath(f"{sp.stem}_{md.dir.name}_rgb-lw.png"),
            )
    del lwflux_rgb

    return agg,disagg,D,C

if __name__=="__main__":
    swath_dir = Path("data/swaths_val")
    model_dir = Path("data/models/")
    fig_dir = Path("figures/swaths")
    eval_dir = Path("data/eval")

    max_swaths = 48
    models = [p for p in model_dir.iterdir() if "ceda-2" in p.name]
    '''
    swaths = list(map(lambda p:swath_dir.joinpath(p), [
        "swath_alk_aqua_20190419-2304.h5",
        "swath_alk_terra_20180217-2215.h5",
        "swath_azn_aqua_20190821-1813.h5",
        "swath_hkh_aqua_20200717-0813.h5",
        "swath_hkh_terra_20181008-0548.h5",
        "swath_idn_aqua_20181129-0510.h5",
        "swath_idn_terra_20200523-0143.h5",
        "swath_neus_aqua_20181020-1743.h5",
        "swath_neus_aqua_20191113-1801.h5",
        "swath_neus_aqua_20200518-1743.h5",
        "swath_neus_terra_20200718-1533.h5",
        "swath_seus_aqua_20180908-1844.h5",
        "swath_seus_terra_20200814-1654.h5",
        ]))
    '''
    swaths = list(swath_dir.iterdir())

    rng = np.random.default_rng(seed=None)
    rng.shuffle(swaths)
    combos = [(m,s) for s in swaths[:max_swaths] for m in models]

    for md,sp in combos:
        eval_str = f"{sp.stem}_{md.name}"
        single_eval_dir = eval_dir.joinpath(eval_str)
        if single_eval_dir.exists():
            print(f"skipping {single_eval_dir}")
            continue
        single_eval_dir.mkdir(exist_ok=False)
        agg,disagg,pred,ceres = eval_swath_grid(sp, md, single_eval_dir)
        pkl.dump((agg, disagg, pred, ceres),
                 single_eval_dir.joinpath(eval_str+".pkl").open("wb"))
        del agg, disagg, pred, ceres
        gc.collect()
