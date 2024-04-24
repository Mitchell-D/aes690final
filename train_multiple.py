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


"""
base_config contains configuration values to all models,
so it should only have fields not subject to variations.
"""
base_config = {
        ## Meta-info
        "model_name":"test-1",
        "model_type":"paed",
        "rand_seed":200007221752,

        ## Exclusive to paed
        "num_modis_bands":36,
        "num_geom_bands":3,
        "num_latent_bands":8,
        "num_ceres_bands":2,

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
        "learning_rate":1e-5,
        "loss":"mse",
        "metrics":["mse", "mae"],
        "weighted_metrics":["mse", "mae"],

        ## Exclusive to train
        "early_stop_metric":"val_mse", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":64,
        "batch_buffer":4,
        "max_epochs":2048, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        "train_val_ratio":.9,

        "notes":"",
        }

## Initialize the model
model,md = ModelDir.build_from_config(
        base_config,
        model_parent_dir=Path("data/models"),
        print_summary=False,
        )

print(model.summary)
print(md)

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
