#%%
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
import numpy as np
import matplotlib.pyplot as plt
from DataHandling.features.slices import read_tfrecords,feature_description,slice_loc
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import predict
from utils.data_decompose import decompse_feature

var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30

file_path = slice_loc(y_plus,var,target,normalized=False)
path_test = os.path.join(file_path,"validation")
print(path_test)

feature_dict = feature_description(file_path)
print(feature_dict)

dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )


target_path = "/home/yuning/thesis/valid/tensors"

numpy_dict = {}
names = list(feature_dict.keys())
names.remove(target[0])
for name in names:
    numpy_dict[name] = []
numpy_dict[target[0]] = []
#%%
for snap in dataset:
    (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
    for name in names:
        value = dict_for_dataset[name].numpy()
        print(type(value))
        numpy_dict[name].append(value)
        print(type(numpy_dict[name][-1]))
    target_array = target_array.numpy()
    numpy_dict[target[0]].append(target_array)
#%%
case_path = os.path.join(target_path,"y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux")
if os.path.exists(case_path) is False:
    os.mkdir(case_path)
    print(f"Made case path {case_path}")


#%%
import torch 
from torch.utils.data import TensorDataset 

for name in numpy_dict.keys():
    tensor_list = [ torch.tensor(i) for i in  numpy_dict[name] ]
    tensors = TensorDataset(torch.stack(tensor_list))
    print(tensors)
    
    torch.save(tensors,case_path+"/{}.pt".format(name))
    print(f"Tensor {name} has been saved!")

# %%
