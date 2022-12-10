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