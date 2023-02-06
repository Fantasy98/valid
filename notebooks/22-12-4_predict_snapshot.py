# 22-12-3 Find the way to extract and decompose the data from given TFRecord dataset
import os
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from DataHandling.features.slices import read_tfrecords,feature_description,load_from_scratch
from DataHandling import utility

from DataHandling.models import models


var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30


file_path = "/home/yuning/thesis/valid/scratch"
path_test = os.path.join(file_path,"test")
print(path_test)
dataset = tf.data.TFRecordDataset(
                                    filenames=path_test,
                                    compression_type="GZIP",
                                    num_parallel_reads=tf.data.experimental.AUTOTUNE
                                    )

feature_dict = feature_description(file_path)
print(feature_dict)

for snap in dataset:
    parsed = tf.io.parse_single_example(snap,feature_dict)
    
    (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
    print(dict_for_dataset.keys())
    print(dict_for_dataset["u_vel"].shape)
    print(target_array.shape)
    break
# (dict_for_dataset,target_array) = read_tfrecords(dataset,feature_dict,target)


# print(dict_for_dataset)
# print(type(target_array))

