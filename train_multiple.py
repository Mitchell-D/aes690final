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

from generate_samples import swaths_dataset,tiles_dataset
from get_modis_swath import modis_band_to_wl
from check_swaths import parse_swath_path

""" Establish normalization bounds based on bulk stats from check_swaths """
from norm_coeffs import modis_norm,ceres_norm,geom_norm
mnorm,cnorm,gnorm = map(dict, (modis_norm, ceres_norm, geom_norm))

"""
Config contains configuration values default to all models.
This config may be added to and some fields may be overwritten downstream.
"""

config = {
        ## Meta-info
        "model_name":"ceda-7",
        "model_type":"ceda",
        "seed":None,

        "num_latent_feats":18,
        "kernel_size":1,
        ## ie: Loss += square_regularization_coeff * outputs ** 2
        "square_regularization_coeff":.4,
        ## If True, the same decoder is applied to both the aggregate
        ## and the gridded latent vectors, otherwise a separate decoder
        ## (or 2) is used to make a prediction for each output
        "share_decoder_weights":True,
        ## Each CERES output will have its own decoder(s)
        "separate_output_decoders":True,

        ## bands 21-25 and 27 have nan values;
        ## striping and noise issues still present with others.
        "modis_feats":(1,4,3,2,26,7,20,28,30,31,33),
        "ceres_feats":("sza","vza"),
        "ceres_labels":("swflux", "lwflux"),

        "enc_conv_filters":[128,128,128],
        "enc_activation":"sigmoid",
        "enc_use_bias":True,
        "enc_kwargs":{},
        "enc_out_kwargs":{},
        "enc_dropout":0.,
        "enc_batchnorm":False,

        "dec_conv_filters":[32,32],
        "dec_activation":"relu",
        "dec_use_bias":True,
        "dec_kwargs":{},
        "dec_out_kwargs":{},
        "dec_dropout":0.,
        "dec_batchnorm":True,
        "dec_out_kwargs":{"use_bias":False},

        ## Exclusive to compile_and_build_dir
        "learning_rate":1e-5,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":64,
        "batch_buffer":2,
        "max_epochs":256, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validations
        "steps_per_epoch":128, ## batches to draw per epoch
        "validation_steps":64, ## batches to draw per validation

        ## Exclusive to generator init
        #"train_val_ratio":.9,
        "mask_val":9999.,
        "modis_grid_size":48,
        "num_tiles_procs":2,
        "num_swath_procs":4,
        "deterministic":False,
        "samples_per_swath":256,
        "block_size":4,
        "buf_size_mb":512,
        ## Substrings constraining swath hdf5s used for traning and validation
        "train_regions":("train"),
        "train_sats":("val"),
        #"train_regions":("alk", "azn", "hkh", "idn", "neus", "seus"),
        #"train_sats":("aqua","terra"),
        #"val_regions":("alk", "azn", "hkh", "idn", "neus", "seus"),
        #"val_sats":("aqua","terra"),
        "val_regions":("train"),
        "val_sats":("val"),

        "notes":"larger latent representation ; moderate sized model ; changes to batch configuration ; trained on weaver dataset",
        }
## Count each of the input types for the generators' init function
config["num_modis_feats"] = len(config["modis_feats"])
config["num_ceres_feats"] = len(config["ceres_feats"])
config["num_ceres_labels"] = len(config["ceres_labels"])

rng = np.random.default_rng(seed=config["seed"])

""" Initialize the training and validation data generators given the config """

## collect normalization bounds from the configured defaults for selected bands
mmean,mstdev = map(
        np.array,zip(*[mnorm[band] for band in config["modis_feats"]]))
gmean,gstdev = map(
        np.array,zip(*[gnorm[band] for band in config["ceres_feats"]]))
cmean,cstdev = map(
        np.array,zip(*[cnorm[band] for band in config["ceres_labels"]]))


'''
"""
Provide the training and validation inputs with a collection of swath hdf5s
by calling generate_samples.swaths_dataset to get a tensorflow dataset.
The swath hdf5s must have been created by get_modis_swath.get_modis_swath
"""
max_train_swaths = 120
max_val_swaths = 60
## select the swath hdf5 files to use for training and validation

train_swath_dir = Path("data/swaths")
val_swath_dir = Path("data/swaths_val")

swath_h5s,train_swath_ids = zip(*[
    (s,parse_swath_path(s, True))
    for s in rng.permuted(list(train_swath_dir.iterdir()))
    if any(sat in s.stem for sat in config["train_sats"])
    and any(region in s.stem for region in config["train_regions"])
    ][:max_train_swaths])
val_h5s,val_swath_ids = zip(*[
    (s,parse_swath_path(s, True))
    for s in rng.permuted(list(val_swath_dir.iterdir()))
    if any(sat in s.stem for sat in config["val_sats"])
    and any(region in s.stem for region in config["train_regions"])
    ][:max_val_swaths])
## save each swath's unique ID as a 3-tuple (region, satellite, epoch_time)
config["swaths_train"] = train_swath_ids
config["swaths_val"] = val_swath_ids
config["modis_feats_norm"] = tuple(mmean),tuple(mstdev)
config["ceres_feats_norm"] = tuple(gmean),tuple(gstdev)
config["ceres_labels_norm"] = tuple(cmean),tuple(cstdev)
config["swath_h5s_train"] = list(map(lambda p:p.as_posix(), swath_h5s))
config["swath_h5s_val"] = list(map(lambda p:p.as_posix(),  val_h5s))
## use the configuration to create training and validation datasets.
tgen = swaths_dataset(swath_h5s=config["swath_h5s_train"], **config)
vgen = swaths_dataset(swath_h5s=config["swath_h5s_val"], **config)
'''


#'''
"""
Provide the training and validation inputs with a collection of tiles hdf5s
by calling generate_samples.tiles_dataset to get a tensorflow dataset.
The tiles hdf5s must have been created by generate_samples.get_tiles_h5
"""
train_substrings = (*config["train_regions"], *config["train_sats"])
val_substrings = (*config["val_regions"], *config["val_sats"])
train_tiles_dir = Path("data/tiles_train")
val_tiles_dir = Path("data/tiles_val")

config["tiles_h5s_train"] = list(map(str,filter(
    lambda tiles_path:any(s in tiles_path.name for s in config["train_sats"]) \
            and any(s in tiles_path.name for s in config["train_regions"]),
    train_tiles_dir.iterdir(),
    )))
config["tiles_h5s_val"] = list(map(str,filter(
    lambda tiles_path:any(s in tiles_path.name for s in config["val_sats"]) \
            and any(s in tiles_path.name for s in config["val_regions"]),
    val_tiles_dir.iterdir(),
    )))

#config["tiles_h5s_val"] = list(map(
#    lambda p:val_tiles_dir.joinpath(p).as_posix(),
#    ("tiles_terra_test_val.h5", "tiles_aqua_test_val.h5")))

tgen = tiles_dataset(tiles_h5s=config["tiles_h5s_train"], **config)
vgen = tiles_dataset(tiles_h5s=config["tiles_h5s_val"], **config)

## Probably just use repeat if you set steps_per_epoch and validation_steps
tgen = tgen.repeat()
vgen = vgen.repeat()

'''
vgen = tiles_dataset(
        tiles_h5s=tiles_h5s_val,
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
           show_shapes=True, show_layer_names=True, expand_nested=True,
           show_layer_activations=True)
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

