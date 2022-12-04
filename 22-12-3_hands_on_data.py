import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
from utils.data_decompose import decompse_feature
from utils.data_decompose import Save_TFdata

os.environ['WANDB_DISABLE_CODE']='True'
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[-1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

file_path = "/home/yuning/thesis/valid/scratch"
path_test = os.path.join(file_path,"test")
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30

data_dict = decompse_feature(file_path,path_test,target)

Ratio = 0.8
Batch_Size = 8
Shuffle = 100
Prefetch = 4
Repeat = 2
save_dir = "/home/yuning/thesis/valid/data"
Save_TFdata(data_dict,Ratio,Batch_Size,Shuffle,Repeat,Prefetch,save_dir)