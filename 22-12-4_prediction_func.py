#%%
# Test to use model predict a single snap shot 
# The model is far more to be trained
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
from utils.prediction import predict
from utils.plots import Plot_2D_snapshots
from utils.metrics import RMS_error, Glob_error
os.environ['WANDB_DISABLE_CODE']='True'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[-1], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass
#%%
model_name = "fresh-glitter-54"
all_path = "/home/yuning/thesis/models/trained/"
model_path = os.path.join(all_path,model_name)
print(model_path)
overwrite = False
var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
# %%
from keras import models
model = models.load_model(model_path)
print(model.summary())

#%%
(preds,targets) = predict(model,y_plus,var,target,normalized=False)
#%%
result_path = "/home/yuning/thesis/valid/results"
model_result_path = os.path.join(result_path,model_name)
if os.path.exists(model_result_path) is False:
  os.mkdir(model_result_path)
pred_data_path = os.path.join(model_result_path,"y_pred.npy")
target_data_path = os.path.join(model_result_path,"y_target.npy")
np.save(pred_data_path,preds)
np.save(target_data_path,targets)
#%%
rms = RMS_error(preds,targets)
globs = Glob_error(preds,targets)
print("RMSE = {}%".format(rms))
print("Glob_Error = {}%".format(globs))
error = np.sqrt(  (np.mean(preds,0)-np.mean(targets,0))**2 )
#%%
fig_path = "/home/yuning/thesis/valid/fig"
fig_path_save = os.path.join(fig_path,model_name)
if os.path.exists(fig_path_save) is False:
   os.mkdir(fig_path_save)
fig_dict = {
            "prediction":np.mean(preds,0),
            "target":np.mean(targets,0),
            "error":error}

for name in fig_dict.keys():
  fig_path_single = os.path.join(fig_path_save,name)
  Plot_2D_snapshots(fig_dict[name],fig_path_single)
  print("Figure of {}  has saved!".format(name))
# fig =plt.figure(0)
# clc = plt.imshow(np.squeeze(pred_pr),cmap = "jet")
# plt.title("Prediction")
# plt.colorbar(clc)

# fig.savefig(os.path.join(fig_path_save,"Prediction"))
# fig =plt.figure(1)
# clc = plt.imshow(target_array,cmap = "jet")
# plt.colorbar(clc)
# plt.title("Origin")
# fig.savefig(os.path.join(fig_path_save,"Target"))

# fig =plt.figure(3)

# clc = plt.imshow(np.sqrt((target_array-np.squeeze(pred_pr))**2),cmap = "jet")
# plt.colorbar(clc)
# plt.title("Error")
# fig.savefig(os.path.join(fig_path_save,"RMSE"))
# %%
