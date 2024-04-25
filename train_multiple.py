"""
Script for randomly searching a user-defined combinatorial graph
of model configurations within the tracktrain framework.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

import tracktrain.model_methods as mm
#from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

from generate_samples import swaths_dataset
from get_modis_swath import modis_band_to_wl
from check_swaths import parse_swath_path

train_swath_dir = Path("data/swaths")
val_swath_dir = Path("data/swaths_val")

""" Establish normalization bounds based on bulk stats from check_swaths """

## (mean, stdev) for each MODIS band, in order of wavelength
modis_norm = dict([
    (8, (0.328, 0.251)), (9, (0.337, 0.245)), (3, (0.260, 0.159)),
    (10, (0.281, 0.179)), (11, (0.231, 0.126)), (12, (0.216, 0.112)),
    (4, (0.234, 0.157)), (1, (0.227, 0.163)), (13, (0.123, 0.039)),
    (14, (0.083, 0.016)), (15, (0.139, 0.051)), (2, (0.077, 0.015)),
    (16, (0.138, 0.030)), (17, (0.299, 0.202)), (18, (0.142, 0.033)),
    (19, (0.265, 0.220)), (5, (0.138, 0.118)), (26, (0.174, 0.140)),
    (6, (0.256, 0.126)), (7, (0.037, 0.058)), (20, (0.608, 0.582)),
    (22, (0.107, 0.065)), (21, (292.095, 12.150)), (23, (285.629, 14.621)),
    (24, (285.396, 14.424)), (25, (281.270, 14.441)), (27, (243.168, 6.450)),
    (28, (259.169, 11.424)), (29, (239.674, 5.964)), (30, (251.481, 8.630)),
    (31, (273.524, 16.258)), (32, (257.556, 11.500)), (33, (273.869, 17.158)),
    (34, (272.713, 17.068)), (35, (256.388, 11.172)), (36, (247.794, 8.360)),
    ])

ceres_norm = dict([ ## currently just normalized with neus values
    ("swflux", (282., 120.9)),
    ("lwflux", (236., 27.4)),
    ])

geom_norm = dict([ ## currently just normalized with neus values
    ("sza", (41.75, 2.078)),
    ("vza", (282.029, 120.906)),
    ("raa", (184.5, 47.8)),
    ])

"""
Config contains configuration values default to all models.
This config may be added to and some fields may be overwritten downstream.
"""

config = {
        ## Meta-info
        "model_name":"test-1",
        "model_type":"paed",
        "random_seed":None,

        "num_latent_feats":8,
        "modis_feats":(8,1,4,3,2,18,5,26,6,7,20,27,28,30,31,33),
        "ceres_feats":("sza", "vza", "raa"),
        "ceres_labels":("swflux", "lwflux"),


        "enc_conv_filters":[64,64,32,16],
        "enc_activation":"gelu",
        "enc_use_bias":True,
        "enc_kwargs":{},
        "enc_out_kwargs":{},
        "enc_dropout":.1,
        "enc_batchnorm":True,

        "dec_conv_filters":[32,32,16,16,8],
        "dec_activation":"gelu",
        "dec_use_bias":True,
        "dec_kwargs":{},
        "dec_out_kwargs":{},
        "dec_dropout":.1,
        "dec_batchnorm":True,

        ## Exclusive to compile_and_build_dir
        "learning_rate":1e-3,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":48,
        "batch_buffer":2,
        "max_epochs":2048, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        #"train_val_ratio":.9,
        "mask_val":9999.,
        "modis_grid_size":48,
        "num_swath_procs":2,
        "samples_per_swath":64,
        "block_size":8,
        "buf_size_mb":512,
        ## Substrings constraining swath hdf5s used for traning and validation
        "train_regions":("neus",),
        "train_sats":("aqua",),
        "val_regions":("neus",),
        "val_sats":("aqua",),

        "notes":"",
        }
## Count each of the input types for the generators' init function
config["num_modis_feats"] = len(config["modis_feats"])
config["num_ceres_feats"] = len(config["ceres_feats"])
config["num_ceres_labels"] = len(config["ceres_labels"])


""" Initialize the training and validation data generators given the config """

## collect normalization bounds from the configured defaults for selected bands
mmean,mstdev = map(
        np.array,zip(*[modis_norm[band] for band in config["modis_feats"]]))
gmean,gstdev = map(
        np.array,zip(*[geom_norm[band] for band in config["ceres_feats"]]))
cmean,cstdev = map(
        np.array,zip(*[ceres_norm[band]for band in config["ceres_labels"]]))

## select the swath hdf5 files to use for training and validation
train_h5s,train_swath_ids = zip(*[
    (s,parse_swath_path(s, True)) for s in train_swath_dir.iterdir()
    if any(sat in s.stem for sat in config["train_sats"])
    and any(region in s.stem for region in config["train_regions"])])
val_h5s,val_swath_ids = zip(*[
    (s,parse_swath_path(s, True)) for s in val_swath_dir.iterdir()
    if any(sat in s.stem for sat in config["val_sats"])
    and any(region in s.stem for region in config["train_regions"])])

## save each swath's unique ID as a 3-tuple (region, satellite, epoch_time)
config["swaths_train"] = train_swath_ids
config["swaths_val"] = val_swath_ids
config["modis_feats_norm"] = tuple(mmean),tuple(mstdev)
config["ceres_feats_norm"] = tuple(gmean),tuple(gstdev)
config["ceres_labels_norm"] = tuple(cmean),tuple(cstdev)

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
