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
import torch 
from torch.utils.data import TensorDataset 
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30

file_path = slice_loc(y_plus,var,target,normalized=False)
path_test = os.path.join(file_path,"test")
print(path_test)

feature_dict = feature_description(file_path)
print(feature_dict)

dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )


target_path = "/home/yuning/thesis/valid/tensors"
case_path = os.path.join(target_path,"y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux")
if os.path.exists(case_path) is False:
    os.mkdir(case_path)
    print(f"Made case path {case_path}")
# numpy_dict = {}
names = list(feature_dict.keys())
for tar in target:
    names.remove(tar)
names.sort()
# for name in names:
#     numpy_dict[name] = []
# for tar in target:
#     numpy_dict[tar] = []
num_snap=0
for i in dataset:
    num_snap +=1 
print(num_snap)
features = []
y = []
#%%
indx = 0; t = 0
from tqdm import tqdm
for snap in tqdm(dataset):
    indx +=1
    (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
    snap_list = []  
    tar_list = []
    for name in names:
        value = dict_for_dataset[name].numpy()
        snap_list.append(torch.from_numpy(dict_for_dataset[name].numpy()))
    snap_tensor = torch.stack(snap_list,dim=0)
    
    target_array = target_array.numpy()
    for tar in target:
        # numpy_dict[tar].append(target_array)
        tar_list.append(torch.from_numpy(target_array))

    tar_tensor = torch.stack(tar_list,dim=0)
    features.append(snap_tensor)
    y.append(tar_tensor)
    if indx % (num_snap//2) == 0:
        t +=1
        features_tensor = TensorDataset(torch.stack(features,dim=0))
        tragets_tensor = TensorDataset(torch.stack(y,dim=0))
        features.clear()
        y.clear()
        torch.save(features_tensor,case_path+"/{}{}.pt".format("features",t))
        torch.save(tragets_tensor,case_path+"/{}{}.pt".format("targets",t))
#%%


# features_tensor = TensorDataset(torch.stack(features,dim=0))
# tragets_tensor = TensorDataset(torch.stack(y,dim=0))
# #%%
# torch.save(features_tensor,case_path+"/{}.pt".format("features"))
# torch.save(tragets_tensor,case_path+"/{}.pt".format("targets"))

#%%


# num_data = len(numpy_dict[names[0]])

# features = []
# y = []
# for indx in tqdm(range(num_data)):
#     snap_list = []  
#     tar_list = [] 
#     for name in names:
#         snap_list.append(torch.from_numpy(numpy_dict[name][indx]))
#         snap_tensor = torch.stack(snap_list,dim=0)
#     # print(snap_tensor.size())
#     features.append(snap_tensor)
#     for tar in target:
#         tar_list.append(torch.from_numpy(numpy_dict[tar][indx]))
#         tar_tensor = torch.stack(tar_list,dim=0)
#         # print(tar_tensor.size())
#     y.append(tar_tensor)
# features_tensor = TensorDataset(torch.stack(features,dim=0))
# tragets_tensor = TensorDataset(torch.stack(y,dim=0))
# #%%
# torch.save(features_tensor,case_path+"/{}.pt".format("features"))
# torch.save(tragets_tensor,case_path+"/{}.pt".format("targets"))

# # %%
