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
from plot_swath import gaussnorm,rgb_norm

from krttdkit.visualize import guitools as gt

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

if __name__=="__main__":
    #swath_dir = Path("data/swaths")
    swath_dir = Path("data/swaths")
    #seed = 200007221752
    seed = None
    rng = np.random.default_rng(seed=seed)

    """ Load the model """
    md = ModelDir(Path("data/models/test-15/"))
    #model = md.load_weights("test-10_002_2.061.weights.h5")
    #model = md.load_weights("test-11_010_2.274.weights.h5")
    #model = md.load_weights("test-12_008_2.046.weights.h5") ## looks good
    #model = md.load_weights("test-12_final.weights.h5")
    #model = md.load_weights("test-13_043_6.213.weights.h5")
    model = md.load_weights("test-15_final.weights.h5")
    ppt(md.config)

    """ """
    swath_paths = list(swath_dir.iterdir())
    rng.shuffle(swath_paths)

    ## load a single swath
    for sp in swath_paths:
        ceres,modis = load_swath(sp)
        m = modis.data(md.config["modis_feats"])[np.newaxis,384:512,384:512,...]
        c = ceres.data(md.config["ceres_labels"])
        ## use average geometry conditions for rough estimate since MODIS
        ## geometryscale is currently messed up.
        #g = np.zeros((*m.shape[:-1],2))
        g = np.zeros((*m.shape[:-1],len(md.config["ceres_feats"])))

        #g[...,1] += 1
        ## Fully express every pixel
        p = np.full((*m.shape[:-1],1), 1, dtype=float)
        p /= float(p.shape[1]*p.shape[2])

        mmean,mstd = zip(*[modis_norm[k] for k in md.config["modis_feats"]])
        gmean,gstd = zip(*[geom_norm[k] for k in md.config["ceres_feats"]])
        cmean,cstd = zip(*[ceres_norm[k] for k in md.config["ceres_labels"]])

        m = (m-mmean)/mstd

        ## convolutional encoder disaggregator
        ## or convolutional-encoder-decoder-aggregator
        model_path = md.dir.joinpath("ceda.keras")
        model.save(model_path)

        ## Create a new model for the
        #paed = tf.keras.Model(model.input, model.get_layer("dec_out").output)
        paed = tf.keras.Model(
                model.input,
                model.get_layer("square_reg").output
                #model.get_layer("dec_out_dec-agg").output
                )

        agg = model((m,g,p)) * np.array(cstd) + np.array(cmean)
        disagg = paed((m,g,p)) * np.array(cstd) + np.array(cmean)

        print(sp)
        print(np.amin(disagg[...,0]),
              np.average(disagg[...,0]),
              np.amax(disagg[...,0]))
        print(np.amin(disagg[...,1]),
              np.average(disagg[...,1]),
              np.amax(disagg[...,1]))
        print(agg)

        tc_idxs = tuple(md.config["modis_feats"].index(l) for l in (1,4,3))
        tc_rgb = rgb_norm(gaussnorm(m[...,tc_idxs], contrast=5, gamma=2))

        gt.quick_render(np.squeeze(tc_rgb))
        gt.quick_render(gt.scal_to_rgb(np.squeeze(disagg)[...,0]))
        gt.quick_render(gt.scal_to_rgb(np.squeeze(disagg)[...,1]))

    exit(0)

    ## select the swath hdf5 files to use for training and validation
    train_h5s,train_swath_ids = zip(*[
        (s,parse_swath_path(s, True)) for s in train_swath_dir.iterdir()
        if any(sat in s.stem for sat in config["train_sats"])
        and any(region in s.stem for region in config["train_regions"])])
    val_h5s,val_swath_ids = zip(*[
        (s,parse_swath_path(s, True)) for s in val_swath_dir.iterdir()
        if any(sat in s.stem for sat in config["val_sats"])
        and any(region in s.stem for region in config["train_regions"])])

    tgen = swaths_dataset(
            swath_h5s=train_h5s,
            grid_size=config["modis_grid_size"],
            num_swath_procs=config["num_swath_procs"],
            samples_per_swath=config["samples_per_swath"],
            block_size=config["block_size"],
            buf_size_mb=config["buf_size_mb"],
            modis_feats=config["modis_feats"],
            ceres_labels=config["ceres_labels"],
            ceres_feats=config["ceres_feats"],
            modis_feats_norm=(mmean,mstdev),
            ceres_feats_norm=(gmean,gstdev),
            ceres_labels_norm=(cmean,cstdev),
            seed=config["random_seed"],
            )
    vgen = swaths_dataset(
            swath_h5s=val_h5s,
            grid_size=config["modis_grid_size"],
            num_swath_procs=config["num_swath_procs"],
            samples_per_swath=config["samples_per_swath"],
            block_size=config["block_size"],
            buf_size_mb=config["buf_size_mb"],
            modis_feats=config["modis_feats"],
            ceres_labels=config["ceres_labels"],
            ceres_feats=config["ceres_feats"],
            modis_feats_norm=(mmean,mstdev),
            ceres_feats_norm=(gmean,gstdev),
            ceres_labels_norm=(cmean,cstdev),
            seed=config["random_seed"],
            )

    """ Initialize the model, and build its directory """
    model,md = ModelDir.build_from_config(
            config,
            model_parent_dir=Path("data/models"),
            print_summary=False,
            )

#'''
    """ optionally generate an image model diagram ; has `pydot` dependency """
    from keras.utils import plot_model
    plot_model(model, to_file=md.dir.joinpath(f"{md.name}.png"),
               show_shapes=True, show_layer_names=True)
#'''

    '''
    """ take a look at the data from the generator as a sanity check """
    from krttdkit.visualize import guitools as gt
    vgen_iter = vgen.repeat().as_numpy_iterator()
    for i in range(32):
        (m,g,p),c = next(vgen_iter)
        print(m.shape, g.shape, p.shape, c.shape)
        print(c)
        #gt.quick_render(m[...,1:4])
        #gt.quick_render(gt.scal_to_rgb(m[...,-2]))
    '''


    """
    Train the model. Expects the following fields to be in config:
    "early_stop_metric","early_stop_patience","save_weights_only",
    "batch_size","batch_buffer","max_epochs","val_frequency",
    """
    best_model = train(
        model_dir_path=md.dir,
        train_config=config,
        compiled_model=model,
        gen_training=tgen,
        gen_validation=vgen,
        )



    exit(0)

    comb_failed = []
    comb_trained = []
    vdata = tuple(map(tuple, vdata))
    comb_shape = tuple(len(v) for v in vdata)
    comb_count = np.prod(np.array(comb_shape))
    for i in range(num_samples):
        ## Get a random argument combination from the configuration
        cur_comb = tuple(np.random.randint(0,j) for j in comb_shape)
        cur_update = {
                vlabels[i]:vdata[i][cur_comb[i]]
                for i in range(len(vlabels))
                }
        cur_update["model_name"] = model_base_name+f"-{i:03}"
        cur_config = {**base_config, **cur_update}
        try:
            ## Build a config dict for the selected current combination

            ## Extract and preprocess the data
            from preprocess import load_WRFSCM_training_data
            X,Y,xlabels,ylabels,y_scales = load_WRFSCM_training_data(wrf_nc_path)

            cur_config["num_inputs"] = X.shape[-1]
            cur_config["num_outputs"] = Y.shape[-1]
            cur_config["input_feats"] = xlabels
            cur_config["output_feats"] = ylabels

            ## Initialize the masking data generators
            gen_train,gen_val = mm.array_to_noisy_tv_gen(
                    X=X,
                    Y=Y,
                    tv_ratio=cur_config.get("train_val_ratio"),
                    noise_pct=cur_config.get("mask_pct"),
                    noise_stdev=cur_config.get("mask_pct_stdev"),
                    mask_val=cur_config.get("mask_val"),
                    feat_probs=cur_config.get("mask_feat_probs"),
                    shuffle=True,
                    dtype=tf.float64,
                    rand_seed=cur_config.get("random_seed"),
                    )
            ## Initialize the model
            model,md = ModelDir.build_from_config(
                    cur_config,
                    model_parent_dir=Path("models"),
                    print_summary=False,
                    )
            best_model = train(
                model_dir_path=md.dir,
                train_config=cur_config,
                compiled_model=model,
                gen_training=gen_train,
                gen_validation=gen_val,
                )
        except Exception as e:
            print(f"FAILED update combination {cur_update}")
            raise e
            #print(e)
            comb_failed.append(cur_comb)
        comb_trained.append(cur_comb)
